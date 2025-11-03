"""
Agente Extractor BBVA - Estados Financieros
"""

import os
import asyncio
from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime
from config import PDF_EXTRACTOR_CONFIG  

class PDFExtractorAgent:
    def __init__(self):
        self.pdf_config = PDF_EXTRACTOR_CONFIG  
        self.agent_name = "PDF_Extractor_Agent"

    async def extract_financial_statements(self):
        """
        Método principal - Compatible con arquitectura async
        """
        try:
            print(f" {self.agent_name} iniciando...")
            
            # Configurar rutas
            input_file = os.path.join(
                self.pdf_config['input_path'],
                self.pdf_config['input_filename']
            )
            
            output_file = os.path.join(
                self.pdf_config['output_path'],
                self.pdf_config['output_filename']
            )
            
            # Crear carpeta output si no existe
            os.makedirs(self.pdf_config['output_path'], exist_ok=True)
            
            # Validar archivo de entrada
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"PDF fuente no encontrado: {input_file}")
            
            # Ejecutar extracción (en hilo separado para no bloquear)
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._extract_pdf_sync, input_file, output_file
            )
            
            # Validar resultado
            if self.pdf_config['validate_extraction']:
                validation = self._validate_extraction(output_file)
                result.update(validation)
            
            print(f" {self.agent_name} completado")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "agent": self.agent_name,
                "timestamp": datetime.now().isoformat()
            }
            print(f" {self.agent_name} falló: {str(e)}")
            return error_result

    def _extract_pdf_sync(self, input_file, output_file):
        """Extracción sincrónica del PDF"""
        # Leer PDF
        reader = PdfReader(input_file)
        total_pages = len(reader.pages)
        
        # Validar páginas disponibles
        max_page_needed = max(self.pdf_config['pages_to_extract'])
        if max_page_needed >= total_pages:
            raise ValueError(f"PDF tiene {total_pages} páginas, necesitas hasta página {max_page_needed + 1}")
        
        # Crear nuevo PDF con solo estados financieros
        writer = PdfWriter()
        pages_extracted = []
        
        for page_index in self.pdf_config['pages_to_extract']:
            page = reader.pages[page_index]
            writer.add_page(page)
            pages_extracted.append(page_index + 1)
        
        # Guardar archivo
        with open(output_file, 'wb') as f:
            writer.write(f)
        
        # Obtener información del resultado
        file_size = os.path.getsize(output_file) / 1024  # KB
        
        return {
            "success": True,
            "agent": self.agent_name,
            "input_file": input_file,
            "output_file": output_file,
            "pages_extracted": pages_extracted,
            "total_pages_extracted": len(pages_extracted),
            "original_pages": total_pages,
            "pages_eliminated": total_pages - len(pages_extracted),
            "file_size_kb": round(file_size, 1),
            "timestamp": datetime.now().isoformat()
        }

    def _validate_extraction(self, output_file):
        """Validar que la extracción incluya los 4 estados financieros"""
        try:
            reader = PdfReader(output_file)
            pages_count = len(reader.pages)
            
            # Validar keywords obligatorias
            required_statements = {
                'balance': False,
                'income': False,
                'cashflow': False,
                'equity': False
            }
            
            for page in reader.pages:
                text = page.extract_text().lower()
                if 'statement of financial position' in text:
                    required_statements['balance'] = True
                if 'statement of comprehensive income' in text:
                    required_statements['income'] = True
                if 'statement of cash flows' in text:
                    required_statements['cashflow'] = True
                if 'statement of changes in equity' in text:
                    required_statements['equity'] = True
            
            all_found = all(required_statements.values())
            
            return {
                "validation": {
                    "pages_count": pages_count,
                    "validation_passed": all_found,
                    "statements_found": required_statements,
                    "missing_statements": [k for k, v in required_statements.items() if not v]
                }
            }
        except Exception as e:
            return {"validation": {"validation_passed": False, "error": str(e)}}


    def is_ready(self):
        """Verificar si el agente está listo para funcionar"""
        try:
            # Verificar dependencias
            import PyPDF2
            
            # Verificar rutas
            input_exists = os.path.exists(self.pdf_config['input_path'])
            output_dir_accessible = os.access(self.pdf_config['output_path'], os.W_OK)
            
            return input_exists and output_dir_accessible
        except ImportError:
            return False

# Para testing independiente
if __name__ == "__main__":
    async def test_extractor():
        agent = PDFExtractorAgent()
        result = await agent.extract_financial_statements()
        print(f"Resultado: {result}")
    
    asyncio.run(test_extractor())
