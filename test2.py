import time
import re
from difflib import SequenceMatcher
from main import query_book_recommendation
from book_titles import BOOKS_BY_CATEGORY

# Sample test cases for different categories
test_cases = [
    {"query": "Can you recommend a mystery book?", "category": "mystery"},
    {"query": "Tell me about psychological thrillers.", "category": "psychological thriller"},
    {"query": "Suggest a classic detective story.", "category": "classic detective mystery"},
    {"query": "Recommend a gothic mystery novel.", "category": "gothic mystery"},
    {"query": "What is a good cozy mystery to read?", "category": "cozy mystery"},
    {"query": "Give me a legal thriller with courtroom drama.", "category": "legal thriller"},
    {"query": "Any good Japanese mystery books?", "category": "japanese mystery"},
    {"query": "Historical crime thrillers I should check out?", "category": "historical crime thriller"},
    {"query": "Best Nordic noir books?", "category": "nordic noir"},
    {"query": "Psychological mystery suggestions?", "category": "psychological mystery"},
    {"query": "Something about suspense or conspiracy?", "category": "suspense"},
    {"query": "Books about detectives?", "category": "detective mystery"},
]

# Function to extract book titles from response using fuzzy match
def extract_titles_from_response(response, category_titles, threshold=0.8):
    found = set()
    for title in category_titles:
        if title.lower() in response.lower():
            found.add(title)
        else:
            for line in response.split("\n"):
                ratio = SequenceMatcher(None, title.lower(), line.strip().lower()).ratio()
                if ratio > threshold:
                    found.add(title)
                    break
    return list(found)

# Evaluation
total_tests = len(test_cases)
passed_tests = 0
total_precision = 0
total_recall = 0
total_f1 = 0
total_response_time = 0

print("\n📊 Starting Evaluation...\n")

for idx, test in enumerate(test_cases, 1):
    query = test["query"]
    category = test["category"]
    category_titles = BOOKS_BY_CATEGORY.get(category, [])

    print(f"\n🔎 Test {idx}: {query}  (Category: {category})")

    start_time = time.time()
    response = query_book_recommendation(query, "llama3.2:latest")
    response = query_book_recommendation(query, "deepseek-r1:1.5b")
    response = query_book_recommendation(query, "qwen2.5:latest")
    duration = time.time() - start_time
    total_response_time += duration

    matched_titles = extract_titles_from_response(response, category_titles)

    passed = len(matched_titles) > 0
    precision = len(matched_titles) / len(category_titles) if category_titles else 0
    recall = len(matched_titles) / len(category_titles) if category_titles else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    passed_tests += 1 if passed else 0
    total_precision += precision
    total_recall += recall
    total_f1 += f1

    print(f"✅ Matched Titles: {matched_titles}" if matched_titles else "❌ No matched titles found")
    print(f"⏱️ Response Time: {duration:.2f}s")
    print(f"📏 Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Final results
avg_precision = total_precision / total_tests
avg_recall = total_recall / total_tests
avg_f1 = total_f1 / total_tests
avg_time = total_response_time / total_tests
accuracy = passed_tests / total_tests * 100

print("\n📊 Final Evaluation")
print(f"✅ Passed: {passed_tests}/{total_tests}")
print(f"📏 Avg Precision: {avg_precision:.4f}")
print(f"📏 Avg Recall: {avg_recall:.4f}")
print(f"🎯 Avg F1 Score: {avg_f1:.4f}")
print(f"⏱️ Avg Response Time: {avg_time:.2f}s")
print(f"🏆 Accuracy: {accuracy:.2f}%")
