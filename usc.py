import sys

class USCInterpreter:
    def __init__(self):
        self.variables = {}
        
    def run(self, script):
        lines = script.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("var "):
                self.handle_variable(line)
            elif line.startswith("function "):
                self.handle_function(line)
            elif line.startswith("print("):
                self.handle_print(line)

    def handle_variable(self, line):
        # Extract variable name and value
        line = line[4:]  # Remove "var "
        name, value = line.split(" = ")
        self.variables[name.strip()] = eval(value.strip().replace('"', "'"))

    def handle_function(self, line):
        # Note: Simplified function handling
        pass  # Expand later for function definitions

    def handle_print(self, line):
        # Extract the expression to print
        expression = line[6:-1]  # Remove "print(" and ")"
        result = self.evaluate_expression(expression)
        print(result)

    def evaluate_expression(self, expression):
        # Replace variable names with their values
        for var in self.variables:
            expression = expression.replace(var, str(self.variables[var]))
        return eval(expression)

def main():
    if len(sys.argv) != 2:
        print("Usage: usc <script_file.usc>")
        return

    filename = sys.argv[1]

    try:
        with open(filename, 'r') as file:
            script = file.read()
            interpreter = USCInterpreter()
            interpreter.run(script)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()