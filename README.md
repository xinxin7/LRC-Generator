# LRC Generator

An automatic lyrics timing generator based on OpenAI Whisper. This tool automatically aligns lyrics with audio files to generate timestamped LRC files.

## Features

- Supports multiple audio formats (MP3, WAV, etc.)
- Uses Whisper for accurate speech recognition
- Automatic lyrics-to-audio alignment
- Generates standard LRC format files
- Supports special characters in lyrics
- Handles repeated lyrics sections
- Provides detailed debugging information

## Requirements

- Python 3.8 or higher
- FFmpeg (for audio processing)
- PyTorch (for Whisper model)

## Installation

1. Install FFmpeg (if not already installed):
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   ```

2. Clone the repository:
   ```bash
   git clone [repository-url]
   cd LRC-Generator
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Basic usage:
```bash
lrc-gen generate --audio "path/to/audio" --lyrics "path/to/lyrics"
```

Example:
```bash
lrc-gen generate --audio "./songs/my_song.mp3" --lyrics "./lyrics/my_lyrics.txt"
```

### Handling Filenames with Spaces or Special Characters

If your filenames contain spaces or special characters, you can:

1. Use quotes:
   ```bash
   lrc-gen generate --audio "./songs/my song.mp3" --lyrics "./lyrics/my lyrics.txt"
   ```

2. Use backslash escaping:
   ```bash
   lrc-gen generate --audio ./songs/my\ song.mp3 --lyrics ./lyrics/my\ lyrics.txt
   ```

## File Format Requirements

### Audio Files
- Supported formats: MP3, WAV, M4A, FLAC, etc.
- High-quality audio recommended for better recognition

### Lyrics Files
- Plain text file (.txt)
- UTF-8 encoding
- One line per lyric
- No timestamps
- Arranged in singing order

Example lyrics file format:
```text
verse 1
First line of lyrics
Second line of lyrics
chorus
This is the chorus
Second line of chorus
```

## Output Files

- Generated LRC file will have the same name as the lyrics file (different extension)
- Location: Same directory as the lyrics file
- Format: Standard LRC format, UTF-8 encoding (with BOM)

Example output:
```text
[ti:Song Title]
[ar:Artist]
[al:Album]
[length:03:45]

[00:01.23]First line of lyrics
[00:05.67]Second line of lyrics
[00:10.89]This is the chorus
```

## Debug Information

The program generates two debug files:
1. `whisper_transcription.txt`: Contains Whisper's speech recognition results
2. `debug_segments.txt`: Contains detailed matching process information

## Important Notes

1. First run will download the Whisper model, requiring internet connection
2. Processing time depends on audio length and chosen model size
3. High-quality audio files recommended for better recognition
4. If lyrics matching is not ideal:
   - Check lyrics text accuracy
   - Ensure correct lyrics order
   - Review debug files for detailed matching process

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Issues and Pull Requests are welcome!
