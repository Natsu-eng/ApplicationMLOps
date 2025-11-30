"""
Script pour identifier les appels de logging problématiques - VERSION CORRIGÉE
"""

import ast
import os

def find_problematic_logging_calls(file_path):
    """Trouve les appels de logging avec kwargs problématiques"""
    try:
        # Essayer différents encodages
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"❌ Impossible de décoder {file_path} avec les encodages testés")
            return []
    
    except Exception as e:
        print(f"❌ Impossible de lire {file_path}: {e}")
        return []

    problematic_calls = []
    
    try:
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Vérifier si c'est un appel à logger.info/error/warning/etc.
                if (isinstance(node.func, ast.Attribute) and 
                    isinstance(node.func.value, ast.Name) and
                    'logger' in node.func.value.id and
                    node.func.attr in ['info', 'error', 'warning', 'debug', 'exception']):
                    
                    # Vérifier les kwargs
                    for keyword in node.keywords:
                        if keyword.arg not in ['exc_info', 'stack_info', 'stacklevel', 'extra']:
                            problematic_calls.append({
                                'line': node.lineno,
                                'arg': keyword.arg,
                                'file': file_path
                            })
    
    except SyntaxError as e:
        print(f"❌ Impossible de parser {file_path}: {e}")
    
    return problematic_calls

# Scanner uniquement les fichiers de notre code source
source_dirs = ['src', 'ui', 'orchestrators', 'utils', 'monitoring']

print("🔍 Recherche des appels de logging problématiques...")

for source_dir in source_dirs:
    if not os.path.exists(source_dir):
        continue
        
    for root, dirs, files in os.walk(source_dir):
        # Ignorer les dossiers __pycache__ et env
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        if 'env' in dirs:
            dirs.remove('env')
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                issues = find_problematic_logging_calls(file_path)
                
                if issues:
                    print(f"\n🚨 PROBLÈMES DANS {file_path}:")
                    for issue in issues:
                        print(f"   Ligne {issue['line']}: argument '{issue['arg']}'")

print("\n✅ Analyse terminée!")