import RAG_Core_2

if __name__ == "__main__":
    # The database should be built once by running build_database.py first
    print("\n--- ASL Sign Recognizer ---")
    
    while True:
        print("\nPlease describe the sign you saw. Press Enter to skip any field.")
        
        handshape = input("Enter Handshape description: ")
        movement = input("Enter Movement description: ")
        location = input("Enter Location description: ")
        orientation = input("Enter Palm Orientation description: ")

        user_filters = {}
        if handshape.strip(): user_filters["Handshape"] = handshape
        if orientation.strip(): user_filters["Orientation"] = orientation
        if location.strip(): user_filters["Location"] = location
        if movement.strip(): user_filters["Movement"] = movement

        # If all fields are empty
        if not user_filters:
            print("\nAssistant: No description provided. Please enter at least one characteristic.")
            continue
            
        # Combined query string
        query_parts = [f"{key}: '{value}'" for key, value in user_filters.items()]
        user_query = ". ".join(query_parts) + "."
        print(f"\nConstructed Query: {user_query}")
        
        # RAG function
        llm_answer = RAG_Core_2.advanced_find_sign(user_query, user_filters)
        print(f"\nAssistant: {llm_answer}")

        again = input("\nSearch for another sign? (yes/no): ")
        if again.lower().strip() != 'yes':
            print("Exiting application. Goodbye!")
            break

