@echo off
setlocal EnableDelayedExpansion

if "%~1"=="" (
    echo ERROR: no source folder specified.
    echo Usage: check_audio_exists.bat [source_folder]
    exit /b 1
)

set "source_folder=%~1"
set "destination_folder=%source_folder%\no_audio_files"

if not exist "%destination_folder%" (
    mkdir "%destination_folder%"
)
set "temp_file=%TEMP%\ffprobe_output.txt"

for %%f in ("%source_folder%\*.mp4") do (
    del "%temp_file%" 2>nul
    ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "%%f" > "%temp_file%" 2>nul

    if not exist "%temp_file%" (
        echo No audio stream found in "%%f"
        move "%%f" "%destination_folder%"
    ) else (
        set /p stream_check=<"%temp_file%"
        echo !stream_check!

        if "!stream_check!"=="aac" (
            echo "Audio stream is AAC in %%f"
        ) else (
            echo No audio stream or non-AAC audio stream found in "%%f"
            move "%%f" "%destination_folder%"
        )
        set "stream_check="
    )
)
