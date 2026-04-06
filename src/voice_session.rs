use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use jni::objects::{GlobalRef, JObject};
use jni::JNIEnv;
use transcribe_rs::onnx::parakeet::{ParakeetParams, TimestampGranularity};
use transcribe_rs::vad::{SileroVad, SmoothedVad, Vad};

use crate::{assets, engine};

/// Number of consecutive silent 30 ms frames that trigger auto-stop.
/// 100 frames × 30 ms = 3 seconds.
// const SILENCE_THRESHOLD_FRAMES: usize = 100;
const SILENCE_THRESHOLD_FRAMES: usize = 66; // 1,980 ms, tuned for better responsiveness without cutting off trailing speech.

/// Per-frame VAD bookkeeping shared between `start_recording` and the stream callback.
struct VadCallbackState {
    vad: SmoothedVad,
    /// Leftover samples that don't fill a complete 480-sample frame yet.
    frame_buf: Vec<f32>,
    /// Set to `true` once the VAD detects the first speech frame.
    speech_detected: bool,
    /// Number of consecutive silent frames since speech was last detected.
    silence_frames: usize,
    /// Guard: fire the JNI callback at most once.
    notified: bool,
    /// True after we've sent the "Listening..." notification for the current speech segment.
    speech_notified: bool,
    /// True after we've sent the "Silence detected" notification for the current silence segment.
    silence_notified: bool,
}

pub struct SendStream(#[allow(dead_code)] pub cpal::Stream);
unsafe impl Send for SendStream {}
unsafe impl Sync for SendStream {}

pub struct VoiceSessionState {
    pub stream: Option<SendStream>,
    pub audio_buffer: Arc<Mutex<Vec<f32>>>,
    pub jvm: Arc<jni::JavaVM>,
    pub target_ref: GlobalRef,
    pub last_level_sent: Arc<Mutex<std::time::Instant>>,
    /// Path to the extracted Silero VAD model; `None` if unavailable.
    pub vad_model_path: Option<PathBuf>,
}

fn notify_status(env: &mut JNIEnv, obj: &JObject, msg: &str) {
    if let Ok(jmsg) = env.new_string(msg) {
        let _ = env.call_method(
            obj,
            "onStatusUpdate",
            "(Ljava/lang/String;)V",
            &[(&jmsg).into()],
        );
    }
}

fn notify_level(env: &mut JNIEnv, obj: &JObject, level: f32) {
    let _ = env.call_method(obj, "onAudioLevel", "(F)V", &[level.into()]);
}

fn notify_text(env: &mut JNIEnv, obj: &JObject, text: &str) {
    if let Ok(jtxt) = env.new_string(text) {
        let _ = env.call_method(
            obj,
            "onTextTranscribed",
            "(Ljava/lang/String;)V",
            &[(&jtxt).into()],
        );
    }
}

fn notify_auto_stop(env: &mut JNIEnv, obj: &JObject) {
    let result = env.call_method(obj, "onAutoStop", "()V", &[]);
    if result.is_err() {
        // Clear pending exception so the JVM doesn't abort
        // (e.g. if the target object doesn't implement onAutoStop)
        let _ = env.exception_clear();
        log::warn!("onAutoStop not available on target object");
    }
}

pub fn init_session(mut env: JNIEnv, target: JObject) -> VoiceSessionState {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

    let vm = env.get_java_vm().expect("Failed to get JavaVM");
    let vm_arc = Arc::new(vm);
    let target_ref = env.new_global_ref(&target).expect("Failed to ref target");

    // Extract Silero VAD model synchronously (fast file copy if not already present).
    let vad_model_path = match assets::extract_vad_model(&mut env, &target) {
        Ok(p) => {
            log::info!("VAD model ready at {:?}", p);
            Some(p)
        }
        Err(e) => {
            log::warn!("VAD model unavailable, auto-stop disabled: {}", e);
            None
        }
    };

    let state = VoiceSessionState {
        stream: None,
        audio_buffer: Arc::new(Mutex::new(Vec::new())),
        jvm: vm_arc.clone(),
        target_ref: target_ref.clone(),
        last_level_sent: Arc::new(Mutex::new(std::time::Instant::now())),
        vad_model_path,
    };

    // Load engine in background
    let vm_clone = vm_arc.clone();
    let target_ref_clone = target_ref.clone();

    std::thread::spawn(move || {
        let _ = engine::ensure_loaded_from_thread(&vm_clone, &target_ref_clone);
    });

    state
}

pub fn start_recording(mut env: JNIEnv, state: &mut VoiceSessionState) {
    let host = cpal::default_host();
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            notify_status(
                &mut env,
                state.target_ref.as_obj(),
                "Error: no microphone available. Check permissions.",
            );
            return;
        }
    };

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    state.audio_buffer.lock().unwrap().clear();
    let buffer_clone = state.audio_buffer.clone();

    let jvm = state.jvm.clone();
    let target_ref = state.target_ref.clone();
    let last_sent = state.last_level_sent.clone();

    // Build Silero VAD if a model is available.
    let vad_state: Option<Arc<Mutex<VadCallbackState>>> =
        state.vad_model_path.as_ref().and_then(|path| {
            match SileroVad::new(path, 0.3) {
                Ok(silero) => {
                    let smoothed = SmoothedVad::new(
                        Box::new(silero),
                        /*prefill_frames=*/ 15,
                        /*hangover_frames=*/ 15,
                        /*onset_frames=*/ 2,
                    );
                    Some(Arc::new(Mutex::new(VadCallbackState {
                        vad: smoothed,
                        frame_buf: Vec::new(),
                        speech_detected: false,
                        silence_frames: 0,
                        notified: false,
                        speech_notified: false,
                        silence_notified: false,
                    })))
                }
                Err(e) => {
                    log::warn!("Failed to init SileroVad, auto-stop disabled: {}", e);
                    None
                }
            }
        });

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            buffer_clone.lock().unwrap().extend_from_slice(data);

            // compute RMS
            let mut sum = 0.0f32;
            for &x in data {
                sum += x * x;
            }
            let rms = (sum / (data.len() as f32)).sqrt();
            let level = (rms * 6.0).clamp(0.0, 1.0);

            // throttle level updates
            let mut last = last_sent.lock().unwrap();
            if last.elapsed() >= std::time::Duration::from_millis(50) {
                *last = std::time::Instant::now();

                if let Ok(mut env) = jvm.attach_current_thread() {
                    let obj = target_ref.as_obj();
                    notify_level(&mut env, obj, level);
                }
            }
            drop(last); // release before VAD section

            // VAD: accumulate into 480-sample frames and check for silence.
            if let Some(vad_arc) = &vad_state {
                let mut should_auto_stop = false;
                let mut should_notify_speech = false;
                let mut should_notify_silence = false;
                {
                    let mut vs = vad_arc.lock().unwrap();
                    if !vs.notified {
                        vs.frame_buf.extend_from_slice(data);
                        while vs.frame_buf.len() >= 480 {
                            let frame: Vec<f32> = vs.frame_buf.drain(..480).collect();
                            if let Ok(is_speech) = vs.vad.is_speech(&frame) {
                                if is_speech {
                                    vs.speech_detected = true;
                                    vs.silence_frames = 0;
                                    if vs.silence_notified {
                                        vs.silence_notified = false;
                                        vs.speech_notified = false;
                                    }
                                    if !vs.speech_notified {
                                        vs.speech_notified = true;
                                        should_notify_speech = true;
                                    }
                                } else if vs.speech_detected {
                                    if vs.speech_notified && !vs.silence_notified {
                                        vs.silence_notified = true;
                                        should_notify_silence = true;
                                    }
                                    vs.silence_frames += 1;
                                    if vs.silence_frames >= SILENCE_THRESHOLD_FRAMES {
                                        vs.notified = true;
                                        should_auto_stop = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                } // release Mutex before JNI call
                if should_notify_speech || should_notify_silence || should_auto_stop {
                    if let Ok(mut env) = jvm.attach_current_thread() {
                        let obj = target_ref.as_obj();
                        if should_notify_speech {
                            notify_status(&mut env, obj, "Listening...");
                        }
                        if should_notify_silence {
                            notify_status(&mut env, obj, "Silence detected");
                        }
                        if should_auto_stop {
                            notify_auto_stop(&mut env, obj);
                        }
                    }
                }
            }
        },
        |e| log::error!("Stream err: {}", e),
        None,
    );

    match stream {
        Ok(s) => {
            s.play().ok();
            state.stream = Some(SendStream(s));
            notify_status(&mut env, state.target_ref.as_obj(), "Speak now");
        }
        Err(e) => {
            notify_status(
                &mut env,
                state.target_ref.as_obj(),
                &format!("Error: failed to open microphone: {}", e),
            );
        }
    }
}

pub fn stop_recording(mut env: JNIEnv, state: &mut VoiceSessionState) {
    // Drop the stream to stop recording
    state.stream = None;

    let buffer = state.audio_buffer.lock().unwrap().clone();

    // Guard against empty buffer (mic permission denied, instant stop, etc.)
    if buffer.is_empty() {
        notify_status(
            &mut env,
            state.target_ref.as_obj(),
            "Error: no audio recorded. Check microphone permissions.",
        );
        return;
    }

    let jvm = state.jvm.clone();
    let target_ref = state.target_ref.clone();

    notify_status(&mut env, target_ref.as_obj(), "Transcribing...");

    std::thread::spawn(move || {
        let mut env = match jvm.attach_current_thread() {
            Ok(e) => e,
            Err(_) => return,
        };
        let obj = target_ref.as_obj();

        // Wait for engine if somehow still loading
        if engine::get_engine().is_none() {
            if let Err(_) = engine::ensure_loaded(&mut env, obj) {
                return;
            }
        }

        if let Some(eng_arc) = engine::get_engine() {
            let res = {
                let mut eng = eng_arc.lock().unwrap();
                let params = ParakeetParams {
                    timestamp_granularity: Some(TimestampGranularity::Segment),
                    ..Default::default()
                };
                eng.transcribe_with(&buffer, &params)
            };

            match res {
                Ok(r) => {
                    notify_status(&mut env, obj, "Ready");
                    notify_text(&mut env, obj, &r.text);
                }
                Err(e) => notify_status(&mut env, obj, &format!("Error: {}", e)),
            }
        } else {
            notify_status(&mut env, obj, "Error: model not loaded");
        }
    });
}

pub fn cancel_recording(mut env: JNIEnv, state: &mut VoiceSessionState) {
    state.stream = None;
    state.audio_buffer.lock().unwrap().clear();
    notify_status(&mut env, state.target_ref.as_obj(), "Canceled");
}
