import os
import json
import argparse
import datetime
import numpy as np
import wfdb

def convert_wfdb_to_jsonl(input_dir, output_file, device_id, session_id, max_records=None):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Find all .hea files in the input directory
    hea_files = [f for f in os.listdir(input_dir) if f.endswith('.hea')]
    hea_files.sort()

    if not hea_files:
        print(f"No .hea files found in {input_dir}")
        return

    print(f"Found {len(hea_files)} records. Starting conversion...")

    if max_records:
        hea_files = hea_files[:max_records]
        print(f"Limiting to first {max_records} records.")

    # Open output file
    with open(output_file, 'w') as f_out:
        for idx, hea_file in enumerate(hea_files, start=1):
            record_name = os.path.splitext(hea_file)[0]
            record_path = os.path.join(input_dir, record_name)
            
            try:
                # Read WFDB record
                record = wfdb.rdrecord(record_path)
                
                # Extract signal data and replace NaNs with 0.0
                signal = record.p_signal
                if signal is None:
                    print(f"Warning: No signal data found in {hea_file}. Skipping.")
                    continue
                
                # Replace NaNs or infinite values to ensure JSON compatibility
                signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Prepare signal list (samples_by_time format: [[lead1, lead2, ...], [lead1, lead2, ...]])
                samples = signal.tolist()
                
                # Compute metadata
                sampling_rate = float(record.fs)
                duration = float(record.sig_len / record.fs)
                
                # Generate IDs
                frame_id = f"{idx:06d}"
                message_id = f"{device_id}-{session_id}-frame_{frame_id}"
                
                # Timestamp matching requested offset format
                created_at = datetime.datetime.now().astimezone().isoformat()
                
                # Construct JSON structure
                record_data = {
                    "message_id": message_id,
                    "device_id": device_id,
                    "session_id": session_id,
                    "frame_id": frame_id,
                    "created_at": created_at,
                    "sampling_rate_hz": sampling_rate,
                    "duration_s": duration,
                    "validation": {
                        "status": "PASS",
                        "warnings": []
                    },
                    "ecg": {
                        "format": "samples_by_time",
                        "samples": samples
                    },
                    "prediction": {
                        "status": "PASS",
                        "label": "Normal",
                        "confidence_percent": 99.73,
                        "probabilities": {
                            "AF": 0.12,
                            "Bradikardia": 0.31,
                            "Normal": 99.73,
                            "Takikardia": 0.07
                        },
                        "threshold": 0.5,
                        "latency_ms": 247.06,
                        "runtime": "ai-edge-litert"
                    },
                    "system": {
                        "cpu_usage_percent": 18.2,
                        "memory_usage_percent": 30.9,
                        "memory_usage_mb": 1170,
                        "cpu_temperature_c": 61.8,
                        "uptime_s": 1402
                    },
                    "stress_test": {
                        "enabled": False,
                        "frame_counter": idx
                    },
                    "network": {
                        "mqtt_publish_latency_ms": 891.6,
                        "wifi_rssi_dbm": -72,
                        "mqtt_connected": True
                    }
                }
                
                # Write to jsonl
                f_out.write(json.dumps(record_data) + '\n')
                print(f"[{idx}/{len(hea_files)}] Converted {record_name} (fs={sampling_rate}Hz, duration={duration}s, channels={record.n_sig})")
                
            except Exception as e:
                print(f"Error processing {record_name}: {e}")

    print(f"Conversion complete. Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WFDB (.hea/.dat) ECG dataset records to JSONL format.")
    parser.add_argument("--input_dir", type=str, default="dataset/ptbxl/sample_100hz",
                        help="Directory containing the WFDB .hea and .dat files.")
    parser.add_argument("--output_file", type=str, default="output/samples.jsonl",
                        help="Path to the output .jsonl file.")
    parser.add_argument("--device_id", type=str, default="device01",
                        help="Device ID to include in the JSON records.")
    parser.add_argument("--session_id", type=str, default=None,
                        help="Session ID to include in the JSON records. Defaults to a timestamp-based ID.")
    parser.add_argument("--max_records", type=int, default=None,
                        help="Maximum number of records to convert.")

    args = parser.parse_args()

    # Generate a default session ID if not provided
    if not args.session_id:
        args.session_id = f"session_{datetime.datetime.now().strftime('%d%m%Y_%H%M%S')}"

    convert_wfdb_to_jsonl(
        input_dir=args.input_dir,
        output_file=args.output_file,
        device_id=args.device_id,
        session_id=args.session_id,
        max_records=args.max_records
    )
