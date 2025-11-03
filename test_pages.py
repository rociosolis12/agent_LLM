from PyPDF2 import PdfReader

# Ajusta esta ruta a tu ubicación real
pdf_path = r"C:\Users\rocio.solis\OneDrive - Accenture\Desktop\Rocio\TFM\data\entrada\input\bbva_2023.pdf"

reader = PdfReader(pdf_path)
print(f"📄 Total de páginas en el documento: {len(reader.pages)}\n")
print("🔍 Buscando estados financieros...\n")

estados_encontrados = {}

for i in range(min(100, len(reader.pages))):
    text = reader.pages[i].extract_text().lower()
    
    if 'statement of financial position' in text and 'as at 31 december' in text:
        if 'notes to' not in text[:300]:
            print(f"✅ BALANCE ENCONTRADO:")
            print(f"   Página del documento: {i+1}")
            print(f"   Índice para PyPDF2: {i}")
            estados_encontrados['balance'] = i
            print()
    
    if 'statement of comprehensive income' in text and 'for the year ended' in text:
        print(f"✅ ESTADO DE RESULTADOS ENCONTRADO:")
        print(f"   Página del documento: {i+1}")
        print(f"   Índice para PyPDF2: {i}")
        estados_encontrados['income'] = i
        print()
    
    if 'statement of changes in equity' in text:
        print(f"✅ CAMBIOS EN PATRIMONIO:")
        print(f"   Página: {i+1}, Índice: {i}")
        estados_encontrados['equity'] = i
        print()
    
    if 'statement of cash flows' in text:
        print(f"✅ FLUJOS DE EFECTIVO:")
        print(f"   Página: {i+1}, Índice: {i}")
        estados_encontrados['cashflow'] = i
        print()

print("=" * 60)
if estados_encontrados:
    indices = list(estados_encontrados.values())
    min_idx = min(indices)
    max_idx = max(indices)
    
    print("RANGO RECOMENDADO PARA config.py:")
    print(f"list(range({min_idx}, {max_idx + 2}))")
else:
    print("⚠️ No se encontraron estados financieros")
print("=" * 60)
