from PyPDF2 import PdfReader, PdfWriter

def remove_pages(pdf_path, output_path):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    # Add all pages, but the selected ones (if you don't want the first 4 then user the number 5 and so on)
    for i in range(5, len(reader.pages)):
        writer.add_page(reader.pages[i])

    with open(output_path, 'wb') as f_out:
        writer.write(f_out)

    print(f"Arquivo salvo: {output_path}")

input_pdf = "PDFs/soup_recipes.pdf"
output_pdf = "PDFs/soup_recipes2.pdf"

remove_pages(input_pdf, output_pdf)
