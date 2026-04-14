package dev.notune.transcribe;

import android.content.res.TypedArray;
import android.graphics.drawable.Drawable;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.LinearLayout;
import android.widget.Switch;
import java.io.File;
import java.io.IOException;

public class MainActivity extends Activity {
    private static final String TAG = "MainActivity";
    private static final int PERM_REQ_CODE = 101;

    static {
        try {
            System.loadLibrary("c++_shared");
            System.loadLibrary("onnxruntime");
        } catch (UnsatisfiedLinkError e) {
            Log.w(TAG, "Failed to load dependencies (c++_shared or onnxruntime)", e);
        }
        System.loadLibrary("android_transcribe_app");
    }

    private static final String THEME_FILE_DARK = "theme_dark";
    private static final String THEME_FILE_BLACK = "theme_black";

    private TextView statusText;
    private Button grantButton;
    private View permsCard;
    private Button startSubsButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        applyTheme();
        setContentView(R.layout.activity_main);
        applyCardThemes();

        statusText = findViewById(R.id.text_status);
        permsCard = findViewById(R.id.card_permissions);
        grantButton = findViewById(R.id.btn_grant_perms);
        startSubsButton = findViewById(R.id.btn_subs_start);
        Button imeSettingsButton = findViewById(R.id.btn_ime_settings);

        grantButton.setOnClickListener(v -> checkAndRequestPermissions());
        
        imeSettingsButton.setOnClickListener(v -> {
             Intent intent = new Intent(Settings.ACTION_INPUT_METHOD_SETTINGS);
             startActivity(intent);
        });

        startSubsButton.setOnClickListener(v -> {
            Intent intent = new Intent(this, LiveSubtitleActivity.class);
            startActivity(intent);
        });

        Switch autoRecordSwitch = findViewById(R.id.switch_auto_record);
        File autoRecordFile = new File(getFilesDir(), "auto_record");
        autoRecordSwitch.setChecked(autoRecordFile.exists());
        autoRecordSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                try {
                    autoRecordFile.createNewFile();
                } catch (IOException e) {
                    Log.e(TAG, "Failed to create auto_record file", e);
                }
            } else {
                autoRecordFile.delete();
            }
        });

        Switch selectTranscriptionSwitch = findViewById(R.id.switch_select_transcription);
        File selectTranscriptionFile = new File(getFilesDir(), "select_transcription");
        selectTranscriptionSwitch.setChecked(selectTranscriptionFile.exists());
        selectTranscriptionSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                try {
                    selectTranscriptionFile.createNewFile();
                } catch (IOException e) {
                    Log.e(TAG, "Failed to create select_transcription file", e);
                }
            } else {
                selectTranscriptionFile.delete();
            }
        });

        Switch pauseAudioSwitch = findViewById(R.id.switch_pause_audio);
        File pauseAudioFile = new File(getFilesDir(), "pause_audio");
        pauseAudioSwitch.setChecked(pauseAudioFile.exists());
        pauseAudioSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                try {
                    pauseAudioFile.createNewFile();
                } catch (IOException e) {
                    Log.e(TAG, "Failed to create pause_audio file", e);
                }
            } else {
                pauseAudioFile.delete();
            }
        });

        // Initial check
        updatePermissionUI();
        
        // Theme selector
        RadioGroup rgTheme = findViewById(R.id.rg_theme);
        File darkFile = new File(getFilesDir(), THEME_FILE_DARK);
        File blackFile = new File(getFilesDir(), THEME_FILE_BLACK);
        if (blackFile.exists()) {
            rgTheme.check(R.id.rb_theme_black);
        } else if (darkFile.exists()) {
            rgTheme.check(R.id.rb_theme_dark);
        } else {
            rgTheme.check(R.id.rb_theme_light);
        }
        rgTheme.setOnCheckedChangeListener((group, checkedId) -> {
            darkFile.delete();
            blackFile.delete();
            try {
                if (checkedId == R.id.rb_theme_dark) {
                    darkFile.createNewFile();
                } else if (checkedId == R.id.rb_theme_black) {
                    blackFile.createNewFile();
                }
            } catch (IOException e) {
                Log.e(TAG, "Failed to save theme", e);
            }
            recreate();
        });

        // Start init
        initNative(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        updatePermissionUI();
    }

    private void updatePermissionUI() {
        boolean hasAudio = checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
        if (hasAudio) {
            permsCard.setVisibility(View.GONE);
        } else {
            permsCard.setVisibility(View.VISIBLE);
        }
    }

    private void checkAndRequestPermissions() {
        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.RECORD_AUDIO}, PERM_REQ_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == PERM_REQ_CODE) {
            updatePermissionUI();
        }
    }

    private void applyTheme() {
        File darkFile = new File(getFilesDir(), THEME_FILE_DARK);
        File blackFile = new File(getFilesDir(), THEME_FILE_BLACK);
        if (blackFile.exists()) {
            setTheme(R.style.AppTheme_Black);
        } else if (darkFile.exists()) {
            setTheme(R.style.AppTheme_Dark);
        }
        // else: default light theme already set via AndroidManifest
    }

    private void applyCardThemes() {
        TypedArray ta = getTheme().obtainStyledAttributes(new int[]{R.attr.cardBg});
        int cardColor = ta.getColor(0, 0xFFFFFFFF);
        ta.recycle();
        int[] cardIds = {R.id.card_permissions, R.id.card_ime, R.id.card_settings, R.id.card_subs};
        for (int id : cardIds) {
            View card = findViewById(id);
            if (card != null) {
                Drawable bg = card.getBackground();
                if (bg != null) bg.setTint(cardColor);
            }
        }
    }

    // Called from Rust
    public void onStatusUpdate(String status) {
        runOnUiThread(() -> {
            statusText.setText("Status: " + status);
            if ("Ready".equals(status)) {
                startSubsButton.setEnabled(true);
            }
        });
    }

    private native void initNative(MainActivity activity);
}
