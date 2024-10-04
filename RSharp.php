<?php

// Helper Functions
function todo($message) {
    throw new Exception("TODO: " . $message);
}

function php7_str_starts_with($haystack, $needle) {
    return strpos($haystack, $needle) === 0;
}

function php7_str_ends_with($haystack, $needle) {
    return substr($haystack, -strlen($needle)) === $needle;
}

// Location class: Represents a location in the source code (file, row, column)
class Loc {
    public $file;
    public $row;
    public $col;

    public function __construct($file, $row, $col) {
        $this->file = $file;
        $this->row = $row;
        $this->col = $col;
    }
}

// Token class: Represents a token in the source code (type, value, location)
class Token {
    public $type;
    public $value;
    public $loc;

    public function __construct($type, $value, $loc) {
        $this->type = $type;
        $this->value = $value;
        $this->loc = $loc;
    }
}

// Lexer class: Tokenizes the source code for RSharp
class Lexer {
    private $code;
    private $file;
    private $pos;
    private $row;
    private $col;

    public function __construct($file, $code) {
        $this->file = $file;
        $this->code = $code;
        $this->pos = 0;
        $this->row = 1;
        $this->col = 1;
    }

    // Check if there's more code to process
    public function is_not_empty() {
        return $this->pos < strlen($this->code);
    }

    // Trim leading whitespace
    public function trim_left() {
        while ($this->is_not_empty() && ctype_space($this->code[$this->pos])) {
            if ($this->code[$this->pos] == "\n") {
                $this->row++;
                $this->col = 1;
            } else {
                $this->col++;
            }
            $this->pos++;
        }
    }

    // Get current location in the code
    public function loc() {
        return new Loc($this->file, $this->row, $this->col);
    }

    // Get the next token from the code
    public function next_token() {
        $this->trim_left();
        if (!$this->is_not_empty()) {
            return null;
        }

        $start_pos = $this->pos;
        $ch = $this->code[$this->pos];
        $loc = $this->loc();

        // Handle numbers
        if (ctype_digit($ch)) {
            while ($this->is_not_empty() && ctype_digit($this->code[$this->pos])) {
                $this->pos++;
                $this->col++;
            }
            $value = substr($this->code, $start_pos, $this->pos - $start_pos);
            return new Token('NUMBER', $value, $loc);
        }

        // Handle identifiers (variables, keywords)
        if (ctype_alpha($ch)) {
            while ($this->is_not_empty() && (ctype_alnum($this->code[$this->pos]) || $this->code[$this->pos] == '_')) {
                $this->pos++;
                $this->col++;
            }
            $value = substr($this->code, $start_pos, $this->pos - $start_pos);
            return new Token('IDENTIFIER', $value, $loc);
        }

        // Handle single-character tokens (e.g., operators)
        $this->pos++;
        $this->col++;
        switch ($ch) {
            case '+': return new Token('PLUS', $ch, $loc);
            case '-': return new Token('MINUS', $ch, $loc);
            case '*': return new Token('STAR', $ch, $loc);
            case '/': return new Token('SLASH', $ch, $loc);
            case '=': return new Token('EQUAL', $ch, $loc);
            case '(': return new Token('LPAREN', $ch, $loc);
            case ')': return new Token('RPAREN', $ch, $loc);
            case '{': return new Token('LBRACE', $ch, $loc);
            case '}': return new Token('RBRACE', $ch, $loc);
            case ';': return new Token('SEMICOLON', $ch, $loc);
            default: todo("Unknown token: $ch");
        }
    }
}

// Example Function Classes
class FuncallStmt {
    public $name;
    public $args;

    public function __construct($name, $args) {
        $this->name = $name;
        $this->args = $args;
    }
}

class RetStmt {
    public $expr;

    public function __construct($expr) {
        $this->expr = $expr;
    }
}

class Func {
    public $name;
    public $body;

    public function __construct($name, $body) {
        $this->name = $name;
        $this->body = $body;
    }
}

// Parsing Functions (Simplified)
function parse_function($lexer) {
    // Example: Parse a simple function
    $token = $lexer->next_token();
    if ($token->type !== 'IDENTIFIER') {
        throw new Exception("Expected function name");
    }

    $func_name = $token->value;
    $lexer->next_token(); // Assume next token is '('
    $lexer->next_token(); // Assume closing ')'
    $lexer->next_token(); // Assume opening '{'

    $body = [];
    while (($token = $lexer->next_token())->type !== 'RBRACE') {
        if ($token->type === 'IDENTIFIER') {
            // Handle function call
            $func_call = new FuncallStmt($token->value, []);
            $body[] = $func_call;
        } elseif ($token->type === 'RETURN') {
            // Handle return statement
            $ret_stmt = new RetStmt($lexer->next_token()->value);
            $body[] = $ret_stmt;
        }
    }

    return new Func($func_name, $body);
}

// Code Generation (Python3 Example)
function generate_python3($func) {
    $code = "def " . $func->name . "():\n";
    foreach ($func->body as $stmt) {
        if ($stmt instanceof FuncallStmt) {
            $code .= "    " . $stmt->name . "()\n";
        } elseif ($stmt instanceof RetStmt) {
            $code .= "    return " . $stmt->expr . "\n";
        }
    }
    return $code;
}

// Main Entry Point
function main($argv) {
    if (count($argv) < 3) {
        echo "Usage: php RSharp.php -target python3 <source-file>\n";
        exit(1);
    }

    $target = $argv[1];
    $file_path = $argv[2];

    if (!file_exists($file_path)) {
        throw new Exception("File not found: $file_path");
    }

    $code = file_get_contents($file_path);
    $lexer = new Lexer($file_path, $code);

    $func = parse_function($lexer);

    switch ($target) {
        case '-target':
            if ($argv[2] === 'python3') {
                echo generate_python3($func);
            } else {
                todo("Only Python3 code generation is supported");
            }
            break;
        default:
            todo("Unknown target platform");
    }
}

main($argv);