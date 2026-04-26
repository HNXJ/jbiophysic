import os
import re

def migrate_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Skip if already migrated or if it's the logging utility itself
    if "get_logger" in content and "logger.info" in content:
        # Still might have remaining prints, so we continue
        pass
    
    if "src/jbiophysic/common/utils/logging.py" in filepath:
        return

    # 1. Add imports if missing
    import_line = "from jbiophysic.common.utils.logging import get_logger"
    logger_line = "logger = get_logger(__name__)"
    
    has_logger = "get_logger" in content
    
    lines = content.splitlines()
    new_lines = []
    import_added = False
    
    for line in lines:
        if not has_logger and not import_added and (line.startswith("import ") or line.startswith("from ")):
            new_lines.append(import_line)
            new_lines.append("")
            new_lines.append(logger_line)
            new_lines.append("")
            import_added = True
        
        # Replace print( with logger.info(
        # Handle cases like print("foo") -> logger.info("foo")
        # But avoid parser.print_help() or other method calls
        if "print(" in line and not re.search(r'\.\w+\.print\(', line):
            # Only match stand-alone print(
            line = re.sub(r'(?<!\.)\bprint\(', 'logger.info(', line)
        
        new_lines.append(line)

    with open(filepath, 'w') as f:
        f.write("\n".join(new_lines) + "\n")

# Target directories
dirs = [
    "src/jbiophysic/models",
    "src/jbiophysic/common",
    "src/jbiophysic/viz"
]

for d in dirs:
    for root, _, files in os.walk(d):
        for file in files:
            if file.endswith(".py"):
                migrate_file(os.path.join(root, file))
