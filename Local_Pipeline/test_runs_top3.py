import json
import RAG_Core_2
import time
from datetime import datetime
import re

def run_evaluation(signs_data, test_mode_fields, report_file):
    """
    Runs an evaluation test for a given mode and writes the results to a report file.
    Checks if the correct word is within the Top 3 returned matches.
    """
    report_header = f"STARTING TOP-3 TEST: Using {len(test_mode_fields)} fields: {', '.join(test_mode_fields)}"
    print("\n" + "="*50)
    print(report_header)
    print("="*50)
    report_file.write("="*50 + "\n")
    report_file.write(report_header + "\n")
    report_file.write("="*50 + "\n\n")

    correct_predictions = 0
    total_predictions = 0
    failed_words = []

    for i, (word, details) in enumerate(signs_data.items()):
        total_predictions += 1
        
        user_filters = {}
        for field in test_mode_fields:
            if field in details and details[field]:
                user_filters[field] = details[field]
        
        if len(user_filters) != len(test_mode_fields):
            print(f"Skipping '{word}': Missing one or more required fields for this test mode.")
            total_predictions -= 1
            continue

        query_parts = [f"{key}: '{value}'" for key, value in user_filters.items()]
        user_query = ". ".join(query_parts) + "."
        
        print(f"Testing word {i+1}/{len(signs_data)}: '{word}'...")

        llm_answer = RAG_Core_2.advanced_find_sign_top_3(user_query, user_filters)

        print(f"Assistant's Top 3: {llm_answer}")
        
        # --- TOP-3 ACCURACY CHECK ---
        cleaned_expected_word = re.sub(r'[^A-Z0-9]', '', word.upper())

        raw_candidates = llm_answer.split(',')
        cleaned_candidates = [re.sub(r'[^A-Z0-9]', '', c.upper()) for c in raw_candidates]
        
        if cleaned_expected_word in cleaned_candidates:
            correct_predictions += 1
            print(f"Result: CORRECT (In Top 3)")
        else:
            failed_words.append({"expected": word, "got": llm_answer, "filters_used": user_filters})
            print(f"Result: INCORRECT")
        
        # Pause
        time.sleep(0.2)

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    # --- Generate Report ---
    summary_lines = [
        f"TEST SUMMARY (TOP-3): {len(test_mode_fields)} fields mode\n",
        f"Total words tested: {total_predictions}\n",
        f"Correct predictions (Top-3): {correct_predictions}\n",
        f"Accuracy: {accuracy:.2f}%\n"
    ]
    
    print("\n" + "-"*50)
    for line in summary_lines:
        print(line, end='')
        report_file.write(line)
    print("-"*50)

    if failed_words:
        failure_header = "\n--- Incorrectly Identified Words ---\n"
        print(failure_header.strip())
        report_file.write(failure_header)
        for failure in failed_words:
            report_line = f"Expected: '{failure['expected']}' | Got: '{failure['got']}'\n"
            
            filters_info = f"  - Attributes Provided ({len(failure['filters_used'])}): {', '.join(failure['filters_used'].keys())}\n"
            report_file.write(report_line)
            report_file.write(filters_info)
    report_file.write("-" * 50 + "\n\n")


if __name__ == "__main__":
    DATA_FILE = 'all_yes_descriptions.json'
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        all_signs_data = json.load(f)

    # Timestamped report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"test_report_top3_{timestamp}.txt"

    with open(report_filename, 'w', encoding='utf-8') as report_file:
        print(f"Starting Top-3 evaluation. Results will be saved to '{report_filename}'")
        report_file.write(f"ASL Sign Recognizer Top-3 Evaluation Report - {timestamp}\n\n")

        # Test 1: Using all 4 descriptions
        test_1_fields = ["Handshape", "Orientation", "Location", "Movement"]
        run_evaluation(all_signs_data, test_1_fields, report_file)

        # Test 2: Using 3 descriptions (omitting Orientation)
        test_2_fields = ["Handshape", "Location", "Movement"]
        run_evaluation(all_signs_data, test_2_fields, report_file)
        
        # Test 3: Using 2 descriptions (Handshape and Movement)
        test_3_fields = ["Handshape", "Movement"]
        run_evaluation(all_signs_data, test_3_fields, report_file)

    print(f"\nAll tests complete. Full report saved to '{report_filename}'")