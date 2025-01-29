import re
import pandas as pd
from pdfminer.high_level import extract_text

# Load PDF text
pdf_path = 'FMVSS.pdf'
text = extract_text(pdf_path)

# Define regex patterns to extract information
section_pattern = r'(S4\.\d(?:\.\d+)*)\s+(.*?)((?=S4\.\d|\Z))'
subsection_pattern = r'(S4\.\d+\.\d+)\s+(.*?)((?=S4\.\d+\.\d+|\Z))'
physical_test_pattern = r'S4\.\d+\.\d+\.\d+'

# Extract data
data = []
for section_match in re.finditer(section_pattern, text, re.DOTALL):
    section_number = section_match.group(1).strip()
    part_name = section_match.group(2).strip()

    # Find subsections within the section
    for sub_match in re.finditer(subsection_pattern, section_match.group(3), re.DOTALL):
        sub_section_number = sub_match.group(1).strip()
        part_mechanism = sub_match.group(2).strip()

        # Check for physical test conditions in the subsection
        physical_test_condition = ""
        if re.search(physical_test_pattern, sub_match.group(0)):
            physical_test_condition = re.search(physical_test_pattern, sub_match.group(0)).group(0).strip()

        # Append to data
        data.append([part_name, part_mechanism, section_number, sub_section_number, physical_test_condition])

# Create DataFrame
df = pd.DataFrame(data, columns=['Part Name', 'Part Mechanism', 'Section number', 'Regulation', 'Physical Test Condition'])

# Display the DataFrame (or save to a CSV file if needed)
print(df)

# Optionally save to CSV
csv_path = 'output.csv'
df.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}")
