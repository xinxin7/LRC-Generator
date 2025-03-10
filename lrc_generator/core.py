"""
Core functionality for LRC generation
"""

import os
import re
import librosa
import numpy as np
from typing import Dict, List, Optional, Union


def generate_lrc_file(
        audio_path: str,
        lyrics_path: str,
        output_path: str,
        metadata: Optional[Dict[str, str]] = None,
        offset: float = 0.0,
) -> str:
    """
    Generate a synchronized lyrics (LRC) file based on the audio file and lyrics text.

    Args:
        audio_path: Path to the audio file (MP3)
        lyrics_path: Path to the lyrics text file
        output_path: Path where the LRC file will be saved
        metadata: Optional dictionary with metadata like title, artist, album
        offset: Time offset in seconds to adjust all timestamps (positive = delay, negative = earlier)

    Returns:
        Path to the generated LRC file
    """
    # Load audio file
    print(f"Loading audio file: {audio_path}")
    y, sr = librosa.load(audio_path)

    # Load lyrics from text file
    print(f"Loading lyrics from: {lyrics_path}")
    with open(lyrics_path, 'r', encoding='utf-8') as f:
        lyrics_text = f.read()

    # Process lyrics (remove empty lines, etc.)
    lyrics = process_lyrics(lyrics_text)
    print(f"Processed {len(lyrics)} lines of lyrics")

    # Calculate song duration
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Song duration: {duration:.2f} seconds")

    # Generate timestamps
    timestamps = generate_timestamps(y, sr, duration, len(lyrics))

    # Apply global offset if provided
    if offset != 0.0:
        timestamps = [max(0, t + offset) for t in timestamps]
        print(f"Applied global offset of {offset:.2f} seconds to all timestamps")

    # Add intelligent adjustments for common patterns
    timestamps = adjust_timestamps_for_patterns(lyrics, timestamps, duration)

    # Write LRC file
    write_lrc_file(output_path, lyrics, timestamps, duration, metadata)
    print(f"LRC file written to: {output_path}")

    return output_path


def adjust_timestamps_for_patterns(lyrics: List[str], timestamps: List[float], duration: float) -> List[float]:
    """
    Apply intelligent adjustments to timestamps based on lyrics patterns and structure.

    Args:
        lyrics: List of lyrics lines
        timestamps: Generated timestamps for each line
        duration: Song duration in seconds

    Returns:
        Adjusted timestamps list
    """
    # Create a copy so we don't modify the original
    adjusted = timestamps.copy()

    # Identify sections (verses, choruses, etc.)
    section_indicators = ['verse', 'chorus', 'bridge', 'intro', 'outro', 'pre-chorus']
    section_indices = []

    for i, line in enumerate(lyrics):
        # Check if this line indicates a section
        if any(indicator in line.lower() for indicator in section_indicators) or line.lower().endswith(':'):
            section_indices.append(i)

    # If we identified sections, make adjustments for better synchronization
    if section_indices:
        print(f"Detected {len(section_indices)} song sections")

        # Ensure a bit of pause before each new section
        for i in section_indices:
            # If not the first section and there's a line before it
            if i > 0 and i < len(adjusted):
                # Add a small gap before new sections (if there's room)
                prev_time = adjusted[i - 1]
                curr_time = adjusted[i]

                # If there's not enough gap already, add one
                min_section_gap = 1.0  # 1 second minimum gap before new sections
                if curr_time - prev_time < min_section_gap:
                    # Push this timestamp (and all subsequent ones) forward
                    gap_to_add = min_section_gap - (curr_time - prev_time)
                    for j in range(i, len(adjusted)):
                        adjusted[j] += gap_to_add

    # Ensure timestamps don't go backwards (monotonic increasing)
    for i in range(1, len(adjusted)):
        if adjusted[i] <= adjusted[i - 1]:
            adjusted[i] = adjusted[i - 1] + 0.01  # Ensure at least a tiny gap

    # Ensure the last timestamp isn't too close to the end
    if adjusted and duration > 0:
        min_ending_gap = duration * 0.05  # 5% of song duration
        if duration - adjusted[-1] < min_ending_gap:
            adjusted[-1] = duration - min_ending_gap

    return adjusted


def process_lyrics(lyrics_text: str) -> List[str]:
    """
    Process raw lyrics text into a list of lines ready for LRC generation.
    Handles section indicators and structuring for better synchronization.

    Args:
        lyrics_text: Raw lyrics text

    Returns:
        List of processed lyrics lines
    """
    # Split by newline
    lines = lyrics_text.strip().split('\n')

    # Remove empty lines and leading/trailing whitespace
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    # Handle common lyrics pattern indicators
    processed_lines = []
    section_indicators = ['verse', 'chorus', 'bridge', 'intro', 'outro', 'pre-chorus', 'hook', 'refrain']

    # Flag to track if we found structured sections
    has_structure = False

    for line in non_empty_lines:
        # Check if this line is a section indicator (ends with : or is a known section name)
        is_section = False
        if line.lower().endswith(':'):
            is_section = True
            has_structure = True
        else:
            for indicator in section_indicators:
                if line.lower() == indicator or line.lower().startswith(f"{indicator}:"):
                    is_section = True
                    has_structure = True
                    break

        # Add the line (section indicators help with synchronization)
        processed_lines.append(line)

    # If we found a structured song (with verses, chorus, etc.), we might want to
    # distribute timestamps differently - this information could be returned and used
    # by the timestamp generation function

    return processed_lines


def generate_timestamps(
        y: np.ndarray,
        sr: int,
        duration: float,
        num_lines: int
) -> List[float]:
    """
    Generate timestamps for lyrics lines.

    Args:
        y: Audio time series
        sr: Sampling rate
        duration: Duration of the audio in seconds
        num_lines: Number of lyrics lines

    Returns:
        List of timestamps (in seconds) for each line
    """
    # Start after an initial delay to account for intro music
    # This helps address the common issue of shifted timestamps
    initial_delay = min(5.0, duration * 0.05)  # 5 seconds or 5% of song, whichever is less

    # Reserve time at the end for outro music
    ending_reserve = duration * 0.05

    # Calculate effective song duration for lyrics
    effective_duration = duration - initial_delay - ending_reserve

    # Use onset detection to find potential starting points for lyrics
    # This is more effective than beat detection for many songs
    try:
        # Get onsets (significant changes in audio that often indicate new lyrics)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr,
                                                  wait=1,  # Wait between consecutive onsets
                                                  pre_avg=1,  # Use longer window for smoother detection
                                                  post_avg=1,
                                                  pre_max=1,
                                                  post_max=1)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Filter onsets to only include those within our effective duration
        valid_onsets = [t for t in onset_times if initial_delay <= t <= (duration - ending_reserve)]

        # If we have enough onsets, use them for timestamp generation
        if len(valid_onsets) >= num_lines:
            # Choose onsets strategically for our lyrics
            if num_lines <= 10:
                # For few lines, space them more evenly
                indices = np.linspace(0, len(valid_onsets) - 1, num_lines).astype(int)
                return [valid_onsets[i] for i in indices]
            else:
                # For many lines, try to cluster timestamps where there's more activity
                # Get energy profile to weight onsets
                energy = librosa.feature.rms(y=y)[0]
                energy_times = librosa.times_like(energy, sr=sr)

                # Find segments with higher energy (likely verses/chorus)
                high_energy_segments = []
                threshold = np.mean(energy) * 1.2  # 20% above mean energy

                for i, e in enumerate(energy):
                    if e > threshold:
                        high_energy_segments.append(energy_times[i])

                if len(high_energy_segments) >= num_lines:
                    # Use high energy points with preference
                    indices = np.linspace(0, len(high_energy_segments) - 1, num_lines).astype(int)
                    return [high_energy_segments[i] for i in indices]
                else:
                    # Fallback to using valid onsets
                    indices = np.linspace(0, len(valid_onsets) - 1, num_lines).astype(int)
                    return [valid_onsets[i] for i in indices]
    except Exception as e:
        # Fall back to even distribution if audio analysis fails
        pass

    # Fallback: distribute evenly across effective duration with initial delay
    return list(np.linspace(initial_delay, initial_delay + effective_duration, num_lines))


def write_lrc_file(
        output_path: str,
        lyrics: List[str],
        timestamps: List[float],
        duration: float,
        metadata: Optional[Dict[str, str]] = None
) -> None:
    """
    Write the LRC file with timestamps and lyrics.

    Args:
        output_path: Path where the LRC file will be saved
        lyrics: List of lyrics lines
        timestamps: List of timestamps for each line
        duration: Duration of the audio in seconds
        metadata: Optional dictionary with metadata like title, artist, album
    """
    with open(output_path, 'w', encoding='utf-8') as lrc_file:
        # Write metadata
        if metadata:
            if 'title' in metadata:
                lrc_file.write(f"[ti:{metadata['title']}]\n")
            if 'artist' in metadata:
                lrc_file.write(f"[ar:{metadata['artist']}]\n")
            if 'album' in metadata:
                lrc_file.write(f"[al:{metadata['album']}]\n")

        # Write length metadata
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        lrc_file.write(f"[length:{minutes:02d}:{seconds:02d}]\n\n")

        # Write timestamped lyrics
        for line, timestamp in zip(lyrics, timestamps):
            minutes = int(timestamp // 60)
            seconds = timestamp % 60
            lrc_file.write(f"[{minutes:02d}:{seconds:05.2f}]{line}\n")


def validate_audio_file(file_path: str) -> bool:
    """
    Validate that the audio file exists and is supported.

    Args:
        file_path: Path to the audio file

    Returns:
        True if the file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False

    # Simple extension check - for a real-world application,
    # you might want to do more thorough validation
    valid_extensions = ['.mp3', '.wav', '.flac', '.ogg']
    _, ext = os.path.splitext(file_path.lower())

    return ext in valid_extensions


def validate_lyrics_file(file_path: str) -> bool:
    """
    Validate that the lyrics file exists and is not empty.

    Args:
        file_path: Path to the lyrics file

    Returns:
        True if the file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return bool(content)  # Check if not empty
    except Exception:
        return False