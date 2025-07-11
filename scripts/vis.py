import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import numpy as np
import re
import argparse
from matplotlib.backends.backend_pdf import PdfPages

def load_signal(filepath, sampling_rate):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find where "Data:" section begins
    for i, line in enumerate(lines):
        if line.strip().lower().startswith('data:'):
            data_start = i + 1
            break
    else:
        raise ValueError("No 'Data:' section found in file.")

    from io import StringIO
    data_str = ''.join(lines[data_start:])
    df = pd.read_csv(StringIO(data_str), sep=';', header=None, engine='python')
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
            try:
                if '-' not in line or ';' not in line:
                    continue
                time_range, duration, event_type, _ = line.strip().split(';')
                start_str, end_str = time_range.strip().split('-')
                start = pd.to_datetime(start_str.strip(), format="%d.%m.%Y %H:%M:%S,%f")

                if not re.search(r'\d{2}\.\d{2}\.\d{4}', end_str):
                    end_str = f"{start.strftime('%d.%m.%Y')} {end_str.strip()}"
                else:
                    end_str = end_str.strip()

                end = pd.to_datetime(end_str, format="%d.%m.%Y %H:%M:%S,%f")

                rows.append((start, end, event_type.strip()))
            except Exception as e:
                print(f"Skipping line due to error: {e}\n{line.strip()}")
                continue
    return pd.DataFrame(rows, columns=['start', 'end', 'event_type'])

def generate_pdf(flow_df, thorac_df, spo2_df, events_df, participant_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{participant_id}_visualization.pdf")

    with PdfPages(pdf_path) as pdf:
        start_time = flow_df.index[0]
        end_time = flow_df.index[-1]
        duration = timedelta(minutes=5)

        while start_time < end_time:
            seg_end = start_time + duration

            fig, axs = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
            fig.suptitle(f"{participant_id} - {start_time} to {seg_end}", fontsize=14)

            axs[0].plot(flow_df[start_time:seg_end].index, flow_df[start_time:seg_end]['value'], label="Nasal Flow")
            axs[0].set_ylabel("Nasal Flow (L/min)")
            axs[0].legend(loc='upper right')

            axs[1].plot(thorac_df[start_time:seg_end].index, thorac_df[start_time:seg_end]['value'], color='orange', label="Thoracic Resp.")
            axs[1].set_ylabel("Resp. Amplitude")
            axs[1].legend(loc='upper right')

            axs[2].plot(spo2_df[start_time:seg_end].index, spo2_df[start_time:seg_end]['value'], color='gray', label="SpO₂")
            axs[2].set_ylabel("SpO₂ (%)")
            axs[2].legend(loc='upper right')

            for _, row in events_df.iterrows():
                if row['end'] >= start_time and row['start'] <= seg_end:
                    for ax in axs:
                        ax.axvspan(max(start_time, row['start']), min(seg_end, row['end']), color='red', alpha=0.3)
                        ax.text(max(start_time, row['start']), ax.get_ylim()[1]*0.9, row['event_type'], fontsize=8, color='red')

            axs[2].set_xlabel("Time")
            axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()

            start_time = seg_end

    print(f"✅ Saved visualization to: {pdf_path}")

def find_file(folder, include_keywords, exclude_keywords=[]):
    for fname in os.listdir(folder):
        lower_name = fname.lower()
        if all(k in lower_name for k in include_keywords) and not any(k in lower_name for k in exclude_keywords):
            return os.path.join(folder, fname)
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate signal visualization for a participant.")
    parser.add_argument('-name', required=True, help="Path to the participant folder (e.g., data/AP03)")
    args = parser.parse_args()

    base_path = args.name
    participant = os.path.basename(base_path)
    output_dir = os.path.join("Visualizations")

    try:
        # Find matching files dynamically
        flow_file   = find_file(base_path, ['flow'], ['event'])
        thorac_file = find_file(base_path, ['thorac']) or find_file(base_path, ['abdominal']) or find_file(base_path, ['movement'])
        spo2_file   = find_file(base_path, ['spo2'])
        events_file = find_file(base_path, ['event'])

        if not all([flow_file, thorac_file, spo2_file, events_file]):
            raise FileNotFoundError("One or more required signal files not found in the folder.")

        # Load signals
        flow = load_signal(flow_file, sampling_rate=32)
        thorac = load_signal(thorac_file, sampling_rate=32)
        spo2 = load_signal(spo2_file, sampling_rate=4)
        events = load_event_annotations(events_file)

        # Generate PDF
        generate_pdf(flow, thorac, spo2, events, participant, output_dir)

    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    main()
