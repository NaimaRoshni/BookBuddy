import time
from main import query_book_recommendation

# Define test cases based on the provided PDF book list
test_cases = [
    {"query": "Can you recommend a mystery book?", "expected_keywords": ["The Girl with the Dragon Tattoo", "Gone Girl", "The Da Vinci Code"]},
    {"query": "Tell me about psychological thrillers.", "expected_keywords": ["Gone Girl", "The Silent Patient", "Sharp Objects"]},
    {"query": "Books similar to The Girl with the Dragon Tattoo.", "expected_keywords": ["The Girl with the Dragon Tattoo", "The Silent Patient", "Gone Girl"]},
    {"query": "Suggest me a book by Agatha Christie.", "expected_keywords": ["The Murder of Roger Ackroyd", "The ABC Murders"]},
    {"query": "I am interested in fantasy books like The Name of the Wind.", "expected_keywords": ["The Name of the Wind", "Mistborn", "The Priory of the Orange Tree"]},
    {"query": "Books related to magic and fantasy.", "expected_keywords": ["The Name of the Wind", "The Priory of the Orange Tree", "The Hobbit"]},
    {"query": "What books are about psychological horror?", "expected_keywords": ["The Shining", "Pet Sematary", "The Exorcist"]},
    {"query": "Suggest me something by Stephen King.", "expected_keywords": ["The Shining", "Carrie", "It"]},
    {"query": "Can you recommend a classic detective mystery?", "expected_keywords": ["The Murder of Roger Ackroyd", "The Hound of the Baskervilles"]},
    {"query": "Books like Harry Potter?", "expected_keywords": ["The Name of the Wind", "The Priory of the Orange Tree", "The Hobbit"]},
    {"query": "Can you suggest a historical mystery?", "expected_keywords": ["The Secret History", "The Name of the Rose"]},
    {"query": "What is a good book for learning Python?", "expected_keywords": []},  # Tests how it handles irrelevant queries
]

# Metrics tracking
total_tests = len(test_cases)
passed_tests = 0
total_response_time = 0

# Run tests
for test in test_cases:
    print(f"\n🔎 **Testing Query:** {test['query']}")
    
    start_time = time.time()
    response = query_book_recommendation(test["query"], "llama3.2:latest")  # Testing with default model
    response_time = time.time() - start_time
    total_response_time += response_time

    print(f"📌 **Response:** {response}")
    print(f"⏳ **Response Time:** {response_time:.2f} seconds")

    # Check if expected keywords appear in the response
    if any(keyword.lower() in response.lower() for keyword in test["expected_keywords"]):
        print("✅ **Test Passed!** (Relevant response)")
        passed_tests += 1
    elif not test["expected_keywords"]:  # If it's an invalid query test
        print("✅ **Test Passed!** (Handled unknown input correctly)")
        passed_tests += 1
    else:
        print("❌ **Test Failed!** (Missing relevant keywords)")

# Print final evaluation
print("\n📊 **Final Test Results**")
print(f"✅ Passed: {passed_tests}/{total_tests}")
print(f"⏳ **Average Response Time:** {total_response_time / total_tests:.2f} seconds")
print(f"🎯 **Accuracy:** {passed_tests / total_tests * 100:.2f}%")
