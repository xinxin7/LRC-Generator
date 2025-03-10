"""
Command-line interface for the LRC generator
"""

import os
import sys
import re
import click
from typing import Dict, Optional

from .core import (
    generate_lrc_file,
    validate_audio_file,
    validate_lyrics_file
)
from .utils import extract_metadata_from_filename, ensure_directory_exists
from .whisper_sync import WhisperLyricsSync


@click.group()
def cli():
    """Generate synchronized lyrics (.lrc) files for your songs."""
    pass


@cli.command()
@click.option(
    '--audio', '-a',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    required=True,
    help='Path to the audio file (MP3, WAV, FLAC, OGG)'
)
@click.option(
    '--lyrics', '-l',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    required=True,
    help='Path to the lyrics text file'
)
@click.option(
    '--output', '-o',
    type=click.Path(file_okay=True, dir_okay=False, writable=True),
    help='Path where the LRC file will be saved (defaults to same location as audio file)'
)
@click.option('--title', '-t', help='Title of the song')
@click.option('--artist', '-r', help='Artist name')
@click.option('--album', '-b', help='Album name')
@click.option('--offset', '-f', type=float, default=0.0,
              help='Time offset in seconds to adjust all timestamps (positive = delay, negative = earlier)')
@click.option('--use-whisper/--no-whisper', default=True,
              help='Use Whisper speech recognition for more accurate sync (default: enabled)')
@click.option('--whisper-model', type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              default='base', help='Whisper model size to use (default: base)')
@click.option('--fallback/--no-fallback', default=True,
              help='Fall back to basic sync if Whisper fails (default: enabled)')
def generate(
        audio: str,
        lyrics: str,
        output: Optional[str] = None,
        title: Optional[str] = None,
        artist: Optional[str] = None,
        album: Optional[str] = None,
        offset: float = 0.0,
        use_whisper: bool = True,
        whisper_model: str = "large",
        fallback: bool = True
):
    """Generate an LRC file from an audio file and lyrics text file."""
    # Validate input files
    if not validate_audio_file(audio):
        click.echo(f"Error: Invalid or unsupported audio file: {audio}", err=True)
        sys.exit(1)

    if not validate_lyrics_file(lyrics):
        click.echo(f"Error: Invalid or empty lyrics file: {lyrics}", err=True)
        sys.exit(1)

    # Set default output path if not provided
    if not output:
        audio_basename = os.path.basename(audio)
        audio_name, _ = os.path.splitext(audio_basename)
        audio_dir = os.path.dirname(audio)
        output = os.path.join(audio_dir, f"{audio_name}.lrc")

    # Ensure output directory exists
    ensure_directory_exists(output)

    # Try to extract metadata from filename if not provided
    metadata: Dict[str, str] = {}
    if not all([title, artist, album]):
        extracted = extract_metadata_from_filename(audio)

        if title:
            metadata['title'] = title
        elif 'title' in extracted:
            metadata['title'] = extracted['title']

        if artist:
            metadata['artist'] = artist
        elif 'artist' in extracted:
            metadata['artist'] = extracted['artist']

        if album:
            metadata['album'] = album
        elif 'album' in extracted:
            metadata['album'] = extracted['album']
    else:
        if title:
            metadata['title'] = title
        if artist:
            metadata['artist'] = artist
        if album:
            metadata['album'] = album

    # Generate the LRC file
    try:
        if use_whisper:
            click.echo(f"Generating Whisper-synced LRC file for {os.path.basename(audio)}...")
            click.echo(f"Using Whisper {whisper_model} model for speech recognition")

            try:
                # Try Whisper-based synchronization
                whisper_sync = WhisperLyricsSync(model_size=whisper_model)
                lrc_path = whisper_sync.generate_lrc(audio, lyrics, output, metadata)

                # Apply offset if specified
                if offset != 0.0:
                    click.echo(f"Applying timestamp offset of {offset:.2f} seconds...")
                    # Read the generated LRC file
                    with open(lrc_path, 'r', encoding='utf-8') as f:
                        lrc_content = f.readlines()

                    # Apply offset to timestamps
                    with open(lrc_path, 'w', encoding='utf-8') as f:
                        for line in lrc_content:
                            if re.match(r'^\[\d+:\d+\.\d+\]', line):
                                # Extract timestamp
                                timestamp_match = re.match(r'^\[(\d+):(\d+\.\d+)\](.*)', line)
                                if timestamp_match:
                                    minutes = int(timestamp_match.group(1))
                                    seconds = float(timestamp_match.group(2))
                                    lyrics_text = timestamp_match.group(3)

                                    # Apply offset
                                    time_seconds = minutes * 60 + seconds + offset
                                    minutes_new = int(time_seconds // 60)
                                    seconds_new = time_seconds % 60

                                    # Write adjusted line
                                    f.write(f"[{minutes_new:02d}:{seconds_new:05.2f}]{lyrics_text}\n")
                            else:
                                f.write(line)

                click.echo(f"✓ Whisper-synced LRC file successfully generated: {lrc_path}")

            except Exception as e:
                if fallback:
                    click.echo(f"Whisper sync failed: {str(e)}")
                    click.echo("Falling back to basic synchronization...")

                    if offset != 0.0:
                        click.echo(f"Applying timestamp offset of {offset:.2f} seconds")

                    lrc_path = generate_lrc_file(audio, lyrics, output, metadata, offset)
                    click.echo(f"✓ LRC file generated using fallback method: {lrc_path}")
                else:
                    # Propagate error if no fallback
                    raise
        else:
            # Use basic synchronization
            click.echo(f"Generating LRC file for {os.path.basename(audio)}...")
            if offset != 0.0:
                click.echo(f"Applying timestamp offset of {offset:.2f} seconds")

            lrc_path = generate_lrc_file(audio, lyrics, output, metadata, offset)
            click.echo(f"✓ LRC file successfully generated: {lrc_path}")

        # Provide help for further adjustments
        click.echo("\nℹ️ If timestamps are still off, try regenerating with an offset:")
        click.echo(f"  - If lyrics appear too early: lrc-gen generate -a \"{audio}\" -l \"{lyrics}\" -f 2.0")
        click.echo(f"  - If lyrics appear too late: lrc-gen generate -a \"{audio}\" -l \"{lyrics}\" -f -2.0")

    except Exception as e:
        click.echo(f"Error generating LRC file: {str(e)}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()