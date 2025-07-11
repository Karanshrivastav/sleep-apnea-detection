import os
import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.signal import butter, filtfilt
import re


def load_signal(filepath, sampling_rate):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Try to find the 'Data:' section
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith('data:'):
            data_start = i + 1
            break

    # If 'Data:' found, parse below it
    if data_start is not None:
        from io import StringIO
        data_str = ''.join(lines[data_start:])
    else:
        # If 'Data:' not found, try to detect first line with timestamp; assume everything before it is metadata
        data_lines = []
        for line in lines:
            if re.match(r'\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2},\d{3}', line.strip()):
                data_lines.append(line)
        if not data_lines:
            raise ValueError("No valid timestamped data lines found.")
        data_str = ''.join(data_lines)

    from io import StringIO
    df = pd.read_csv(StringIO(data_str), sep=';', header=None, engine='python')
    df = df.iloc[:, :2]  # Handle files with more than two columns
    df.columns = ['timestamp', 'value']

    df['timestamp'] = pd.to_datetime(df['timestamp'].str.strip(), format="%d.%m.%Y %H:%M:%S,%f")
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.dropna(inplace=True)
    df.set_index('timestamp', inplace=True)
    return df


def load_event_annotations(event_file):
    rows = []
    with open(event_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match lines with format like: 29.05.2024 21:33:57,246-21:34:33,496; 36;Body event; Wake
            match = re.match(
                r'^(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2},\d{3})-(\d{2}:\d{2}:\d{2},\d{3});\s*(\d+);([^;]+);([^;]+)',
                line
            )

            if not match:
                continue  # Skip non-matching or malformed lines

            try:
                start_str, end_str, duration, event_type, sleep_stage = match.groups()

                # Parse start time
                start = pd.to_datetime(start_str, format="%d.%m.%Y %H:%M:%S,%f")
                
                # For end time, use the same date as start time
                end_date = start.strftime('%d.%m.%Y')
                end_full = f"{end_date} {end_str.strip()}"
                end = pd.to_datetime(end_full, format="%d.%m.%Y %H:%M:%S,%f")

                rows.append((start, end, event_type.strip(), sleep_stage.strip()))

            except Exception as e:
                print(f"⚠️ Skipping malformed line in {os.path.basename(event_file)}:\n{line}\nReason: {e}")
                continue

    return pd.DataFrame(rows, columns=['start', 'end', 'event_type', 'sleep_stage'])


def load_sleep_profile(profile_path):
    """Load sleep stage data from profile file"""
    stages = []
    with open(profile_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ';' not in line:
                continue
            try:
                timestamp, stage = line.split(';')
                stages.append(stage.strip())
            except Exception as e:
                print(f"⚠️ Skipping malformed line in sleep profile: {line}\nReason: {e}")
    return stages


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def window_signal(signal_df, window_size_sec=30, overlap=0.5, fs=32):
    window_stride = int((1 - overlap) * window_size_sec * fs)
    window_len = int(window_size_sec * fs)
    signal = signal_df['value'].to_numpy()
    timestamps = signal_df.index.to_numpy()

    windows = []
    time_ranges = []

    for start in range(0, len(signal) - window_len + 1, window_stride):
        end = start + window_len
        window = signal[start:end]
        start_time = timestamps[start]
        end_time = timestamps[end - 1]
        windows.append(window)
        time_ranges.append((start_time, end_time))
    return windows, time_ranges


def assign_labels(time_ranges, events_df):
    labels = []
    sleep_stages = []
    for start, end in time_ranges:
        overlap_events = events_df[
            (events_df['end'] >= start) & (events_df['start'] <= end)
        ]
        matched_label = 'Normal'
        matched_stage = 'Unknown'

        for _, row in overlap_events.iterrows():
            overlap_start = max(start, row['start'])
            overlap_end = min(end, row['end'])
            overlap_duration = (overlap_end - overlap_start) / np.timedelta64(1, 's')
            window_duration = (end - start) / np.timedelta64(1, 's')

            if overlap_duration / window_duration >= 0.5:
                if row['event_type'] in ['Hypopnea', 'Obstructive Apnea']:
                    matched_label = row['event_type']
                matched_stage = row['sleep_stage']
                break

        labels.append(matched_label)
        sleep_stages.append(matched_stage)
    
    return labels, sleep_stages


def find_file(key, participant_path, participant_id):
    key = key.lower()
    files = [f for f in os.listdir(participant_path) if f.lower().endswith('.txt')]
    
    # Special handling for different participant folders
    if participant_id == 'AP04':
        if key == 'flow':
            # AP04 has "Flow Signal" instead of just "Flow"
            for f in files:
                if 'flow' in f.lower() and 'signal' in f.lower() and 'event' not in f.lower():
                    return os.path.join(participant_path, f)
        elif key == 'event':
            for f in files:
                if 'flow' in f.lower() and 'event' in f.lower():
                    return os.path.join(participant_path, f)
        elif key == 'sleep profile':
            for f in files:
                if 'sleep' in f.lower() and 'profile' in f.lower():
                    return os.path.join(participant_path, f)
    
    elif participant_id == 'AP05':
        if key == 'flow':
            # AP05 has "Flow Nasal"
            for f in files:
                if 'flow' in f.lower() and 'nasal' in f.lower() and 'event' not in f.lower():
                    return os.path.join(participant_path, f)
        elif key == 'event':
            for f in files:
                if 'flow' in f.lower() and 'event' in f.lower():
                    return os.path.join(participant_path, f)
        elif key == 'sleep profile':
            for f in files:
                if 'sleep' in f.lower() and 'profile' in f.lower():
                    return os.path.join(participant_path, f)
    
    else:  # For AP01, AP02, AP03
        if key == 'flow':
            for f in files:
                if 'flow' in f.lower() and 'event' not in f.lower():
                    return os.path.join(participant_path, f)
        elif key == 'event':
            for f in files:
                if 'flow' in f.lower() and 'event' in f.lower():
                    return os.path.join(participant_path, f)
        elif key == 'sleep profile':
            for f in files:
                if 'sleep' in f.lower() and 'profile' in f.lower():
                    return os.path.join(participant_path, f)
    
    raise FileNotFoundError(f"Could not find {key} file in {participant_id}")


def process_participant(participant_path, breathing_data, sleep_stage_data):
    participant_id = os.path.basename(participant_path)
    print(f"📦 Processing: {participant_id}")

    try:
        # Find and load all required files
        flow_path = find_file('flow', participant_path, participant_id)
        event_path = find_file('event', participant_path, participant_id)
        profile_path = find_file('sleep profile', participant_path, participant_id)

        # Process breathing data
        flow_df = load_signal(flow_path, sampling_rate=32)
        events_df = load_event_annotations(event_path)
        sleep_stages = load_sleep_profile(profile_path)

        # Filter signal
        filtered_values = bandpass_filter(flow_df['value'], 0.17, 0.4, fs=32)
        flow_df['value'] = filtered_values

        # Window the signal
        windows, time_ranges = window_signal(flow_df, fs=32)
        labels, window_sleep_stages = assign_labels(time_ranges, events_df)

        # Add to breathing dataset
        for i, window in enumerate(windows):
            breathing_data.append({
                'participant_id': participant_id,
                'window_id': f"{participant_id}_{i}",
                **{f'feature_{j}': val for j, val in enumerate(window)},
                'label': labels[i],
                'sleep_stage': window_sleep_stages[i]
            })

        # Add to sleep stage dataset
        sleep_stage_data.append({
            'participant_id': participant_id,
            'sleep_stages': ' '.join(sleep_stages),
            'num_stages': len(sleep_stages)
        })

    except FileNotFoundError as e:
        print(f"❌ File error in {participant_id}: {e}")
    except Exception as e:
        print(f"❌ Processing failed for {participant_id}: {str(e)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', type=str, required=True, help="Path to data directory")
    parser.add_argument('-out_dir', type=str, required=True, help="Path to output dataset directory")
    args = parser.parse_args()

    # Initialize combined datasets
    breathing_data = []
    sleep_stage_data = []

    # Process each participant
    for folder_name in sorted(os.listdir(args.in_dir)):
        full_path = os.path.join(args.in_dir, folder_name)
        if os.path.isdir(full_path):
            process_participant(full_path, breathing_data, sleep_stage_data)

    # Create DataFrames
    breathing_df = pd.DataFrame(breathing_data)
    sleep_stage_df = pd.DataFrame(sleep_stage_data)

    # Save combined datasets
    os.makedirs(args.out_dir, exist_ok=True)
    breathing_df.to_csv(os.path.join(args.out_dir, 'breathing_dataset.csv'), index=False)
    sleep_stage_df.to_csv(os.path.join(args.out_dir, 'sleep_stage_dataset.csv'), index=False)

    print(f"✅ Saved combined datasets to {args.out_dir}")
    print(f"  - Breathing dataset: {len(breathing_df)} windows")
    print(f"  - Sleep stage dataset: {len(sleep_stage_df)} participants")


if __name__ == '__main__':
    main()