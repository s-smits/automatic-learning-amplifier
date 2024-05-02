import os
import re
from stdlib_list import stdlib_list

def get_imports_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    imports = set(re.findall(r'^\s*(?:import|from)\s+(\S+)', content, re.MULTILINE))
    return {imp.split('.')[0] for imp in imports}

def filter_standard_libs(imports):
    standard_libs = set(stdlib_list("3.8"))  # Specify your Python version
    return {imp for imp in imports if imp not in standard_libs}

def resolve_package_names(imports):
    mapping = {"PIL": "Pillow", "sklearn": "scikit-learn"}
    return {mapping.get(imp, imp) for imp in imports}

def generate_requirements(project_path):
    all_imports = set()
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                imports = get_imports_from_file(file_path)
                all_imports.update(imports)
    
    filtered_imports = filter_standard_libs(all_imports)
    final_imports = resolve_package_names(filtered_imports)
    
    with open('requirements.txt', 'w') as req_file:
        for imp in sorted(final_imports):
            req_file.write(f"{imp}\n")

# Usage
generate_requirements('src')