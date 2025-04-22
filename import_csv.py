import csv

def format_authors_harvard(authors_string):
    authors_list = authors_string.split(',')
    formatted_authors = []
    for author in authors_list:
        author = author.strip()
        names = author.split()
        if not names:
            continue  # Handle empty author names if any
        surname = names[-1]
        initials = ''.join([name[0].upper() + '.' for name in names[:-1]])
        formatted_authors.append(f"{surname}, {initials}")
    return ', '.join(formatted_authors)

def create_harvard_bibliography(csv_file_path):
    harvard_bibliography_entries = []
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            title = row['Title'].strip('"')
            authors = row['Authors']
            year = row['Year']
            venue = row['Venue']
            doi_link = row['DOI link']

            formatted_authors = format_authors_harvard(authors)

            venue_lower = venue.lower()
            if any(keyword in venue_lower for keyword in ['conference', 'meeting', 'workshop', 'proceedings', 'symposium']):
                # Conference paper format
                entry = f"{formatted_authors} ({year}) '{title}', in *{venue}*. {doi_link}"
            else:
                # Journal article format
                entry = f"{formatted_authors} ({year}) '{title}'. *{venue}*. {doi_link}"

            harvard_bibliography_entries.append(entry)

    return harvard_bibliography_entries

if __name__ == "__main__":
    csv_file = '/Users/simoneabbiati/Desktop/BoE_Paper/Bibliography.csv'  # Replace with your CSV file path
    bibliography = create_harvard_bibliography(csv_file)

    for entry in bibliography:
        print(entry)