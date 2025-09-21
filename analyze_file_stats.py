import os


def analyze_character_counts(directory: str):
    """
    Analyzes all .txt files in a given directory and its subdirectories,
    calculating the character count for each file and the overall average.

    Args:
        directory (str): The path to the directory to be analyzed.
    """
    if not os.path.isdir(directory):
        print(
            f"Error: The directory '{directory}' was not found or is not a "
            "valid directory."
        )
        return

    file_character_counts = {}
    total_characters = 0
    total_files = 0

    print(f"🔍 Analyzing .txt files in '{directory}'...\n")

    # os.walk is ideal for recursively traversing the directory tree
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                try:
                    # Opens the file with the correct encoding and reads the content
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        char_count = len(content)

                        # Stores the count for the file using the relative path
                        relative_path = os.path.relpath(file_path, directory)
                        file_character_counts[relative_path] = char_count

                        # Updates the totals
                        total_characters += char_count
                        total_files += 1

                except Exception as e:
                    print(f"  - Error reading file {file_path}: {e}")

    if total_files == 0:
        print("No .txt files were found in the directory.")
        return

    # Calculates the average
    average_characters = total_characters / total_files

    # Sorts the files by character count for display
    sorted_files = sorted(
        file_character_counts.items(), key=lambda item: item[1], reverse=True
    )

    # Imprime o resumo final
    print("\n" + "=" * 50)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total .txt files processed: {total_files}")
    print(f"Total characters (sum of all): {total_characters:,}")
    print(f"Average characters per file: {average_characters:,.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # The default path to the folder containing the .txt files
    target_directory = r"C:\Users\igor-\Desktop\godot_project\godot_docs\pages"
    analyze_character_counts(target_directory)