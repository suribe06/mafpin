import pdfplumber

files = [
    "/Users/cero/mafpin/articles/ws-dmaa.pdf",
    "/Users/cero/mafpin/articles/Local_Pluralistic_Homophily_for_Boundary_Spanning_Node_Detection_in_Overlapping_Community_Networks v2.pdf",
]

for fname in files:
    print(f"\n\n{'='*80}")
    print(f"ARTICULO: {fname.split('/')[-1]}")
    print(f"{'='*80}")
    with pdfplumber.open(fname) as pdf:
        total = len(pdf.pages)
        print(f"Total páginas: {total}")
        for i, page in enumerate(pdf.pages[:10]):
            text = page.extract_text() or ""
            if text.strip():
                print(f"\n--- pagina {i+1} ---")
                print(text[:3500])
