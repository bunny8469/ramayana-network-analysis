import re

def getRawText():
    with open("dataset/ramayan_trunc.txt", "r", encoding="utf-8") as f:
        return f.read()

def extract_books_and_cantos(num_books=None, cantos_per_book=None):
    text = getRawText()
    book_pattern = re.compile(
        r"BOOK\s+([IVXLC]+)\.?\s*(.*?)(?=BOOK\s+[IVXLC]+\.?|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    book_matches = book_pattern.findall(text)
    print(f"Found {len(book_matches)} books in the text")

    if num_books:
        book_matches = book_matches[:num_books]
        print(f"Analyzing first {num_books} books")

    all_cantos = []

    def roman_to_int(roman):
        vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
        total, prev = 0, 0
        for ch in reversed(roman.upper()):
            v = vals.get(ch, 0)
            total += -v if v < prev else v
            prev = v
        return total

    for book_roman, book_text in book_matches:
        book_num = roman_to_int(book_roman)
        canto_pattern = re.compile(
            r"Canto\s+([IVXLC]+)\.\s*(.*?)(?=Canto\s+[IVXLC]+\.|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        canto_matches = canto_pattern.findall(book_text)
        print(f"  Book {book_roman} ({book_num}): Found {len(canto_matches)} cantos")

        if cantos_per_book:
            canto_matches = canto_matches[:cantos_per_book]

        for canto_roman, canto_text in canto_matches:
            canto_num = roman_to_int(canto_roman)
            canto_text_clean = canto_text.replace("\n", " ")
            all_cantos.append((book_num, canto_num, canto_text_clean))
    return all_cantos
