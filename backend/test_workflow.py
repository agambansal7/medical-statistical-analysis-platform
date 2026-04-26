#!/usr/bin/env python3
"""Comprehensive test script for the Statistical Analysis Platform."""

import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000/api/v1"
TEST_DATA_PATH = "/Users/agam/Downloads/tavr_racial_disparities_1000patients.csv"

def test_complete_workflow():
    """Test the complete workflow: session -> upload -> research question."""
    print("=" * 60)
    print("STATISTICAL ANALYSIS PLATFORM - WORKFLOW TEST")
    print("=" * 60)

    # Step 1: Health check
    print("\n1. Health Check...")
    try:
        r = requests.get(f"{BASE_URL.replace('/api/v1', '')}/health", timeout=5)
        if r.status_code == 200:
            print(f"   ✓ Server is healthy: {r.json()}")
        else:
            print(f"   ✗ Health check failed: {r.status_code}")
            return
    except Exception as e:
        print(f"   ✗ Server not responding: {e}")
        return

    # Step 2: Create session
    print("\n2. Creating Session...")
    try:
        r = requests.post(f"{BASE_URL}/session/create", timeout=10)
        if r.status_code == 200:
            session_data = r.json()
            session_id = session_data.get("session_id")
            print(f"   ✓ Session created: {session_id}")
        else:
            print(f"   ✗ Session creation failed: {r.status_code} - {r.text}")
            return
    except Exception as e:
        print(f"   ✗ Session creation error: {e}")
        return

    # Step 3: Upload data
    print("\n3. Uploading Data...")
    try:
        with open(TEST_DATA_PATH, 'rb') as f:
            files = {'file': (TEST_DATA_PATH.split('/')[-1], f, 'text/csv')}
            # Pass session_id as query parameter, not form data
            r = requests.post(f"{BASE_URL}/data/upload?session_id={session_id}", files=files, timeout=30)

        if r.status_code == 200:
            upload_result = r.json()
            print(f"   ✓ Data uploaded successfully")
            # Response has 'profile' key, not 'data_profile'
            profile = upload_result.get("profile", {})
            print(f"   - Message: {upload_result.get('message', 'N/A')}")
            print(f"   - Rows: {profile.get('n_rows', 'N/A')}")
            print(f"   - Columns: {profile.get('n_columns', 'N/A')}")
            variables = profile.get('variables', [])
            print(f"   - Variables ({len(variables)}): {[v.get('name') for v in variables[:10]]}...")
        else:
            print(f"   ✗ Data upload failed: {r.status_code}")
            print(f"   Response: {r.text[:500]}")
            return
    except FileNotFoundError:
        print(f"   ✗ Test data file not found: {TEST_DATA_PATH}")
        return
    except Exception as e:
        print(f"   ✗ Data upload error: {e}")
        return

    # Step 4: Ask research question
    print("\n4. Asking Research Question...")
    question = "What is the impact of race and sex on 30-day mortality in TAVR patients?"
    try:
        r = requests.post(
            f"{BASE_URL}/chat/research-question",
            json={"session_id": session_id, "question": question},
            timeout=120  # LLM calls can take time
        )

        if r.status_code == 200:
            result = r.json()
            if result.get("success"):
                plan = result.get("plan", {})
                print(f"   ✓ Research question analyzed!")
                print(f"   - Research type: {plan.get('research_type', 'N/A')}")
                print(f"   - Primary analyses: {len(plan.get('primary_analyses', []))}")
                print(f"   - Secondary analyses: {len(plan.get('secondary_analyses', []))}")
                print(f"   - Validated: {plan.get('validated', False)}")
                print(f"   - Requires confirmation: {result.get('require_confirmation', False)}")

                if plan.get("variable_warnings"):
                    print(f"   - Variable warnings: {plan['variable_warnings'][:5]}")

                # Show primary analyses
                print("\n   PRIMARY ANALYSES:")
                for i, analysis in enumerate(plan.get("primary_analyses", [])[:5], 1):
                    print(f"   {i}. {analysis.get('test_name')}")
                    print(f"      Category: {analysis.get('category')}")
                    print(f"      Rationale: {analysis.get('rationale', '')[:100]}...")
            else:
                print(f"   ✗ Analysis failed: {result.get('error', 'Unknown error')}")
                if result.get("raw_response"):
                    print(f"   Raw response: {result['raw_response'][:300]}...")
        else:
            print(f"   ✗ Research question failed: {r.status_code}")
            print(f"   Response: {r.text[:500]}")
    except requests.exceptions.Timeout:
        print(f"   ✗ Request timed out (LLM may be slow)")
    except Exception as e:
        print(f"   ✗ Research question error: {e}")
        return

    # Step 5: Confirm plan
    print("\n5. Confirming Analysis Plan...")
    try:
        r = requests.post(
            f"{BASE_URL}/chat/confirm-plan/{session_id}",
            json={},  # No modifications
            timeout=10
        )

        if r.status_code == 200:
            result = r.json()
            print(f"   ✓ Plan confirmed!")
            print(f"   - Message: {result.get('message', 'N/A')}")
        else:
            print(f"   ✗ Plan confirmation failed: {r.status_code}")
            print(f"   Response: {r.text[:500]}")
    except Exception as e:
        print(f"   ✗ Plan confirmation error: {e}")

    # Step 6: Execute plan
    print("\n6. Executing Analysis Plan...")
    try:
        r = requests.post(
            f"{BASE_URL}/chat/execute-plan/{session_id}",
            timeout=300  # Analyses can take time
        )

        if r.status_code == 200:
            result = r.json()
            print(f"   ✓ Plan executed!")
            print(f"   - Success: {result.get('success', False)}")
            print(f"   - Executed: {result.get('n_executed', 0)} of {result.get('n_total', 0)}")
            print(f"   - Message: {result.get('message', 'N/A')}")

            # Show results summary
            results = result.get("results", [])
            if results:
                print("\n   RESULTS SUMMARY:")
                for i, res in enumerate(results[:5], 1):
                    test_name = res.get("test_name", "Unknown")
                    success = res.get("result", {}).get("success", False)
                    status = "✓" if success else "✗"
                    print(f"   {i}. {status} {test_name}")

                    if success and res.get("result", {}).get("interpretation"):
                        interp = res["result"]["interpretation"][:200]
                        print(f"      {interp}...")

            # Show errors if any
            errors = result.get("errors", [])
            if errors:
                print("\n   ERRORS:")
                for err in errors:
                    print(f"   - {err.get('test_name')}: {err.get('error')}")
        else:
            print(f"   ✗ Plan execution failed: {r.status_code}")
            print(f"   Response: {r.text[:500]}")
    except requests.exceptions.Timeout:
        print(f"   ✗ Request timed out (analyses may still be running)")
    except Exception as e:
        print(f"   ✗ Plan execution error: {e}")

    print("\n" + "=" * 60)
    print("WORKFLOW TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_complete_workflow()
