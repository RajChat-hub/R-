<?php

// Utility functions
function todo($message) {
    throw new ErrorException("TODO: " . $message);
}

function php7_str_starts_with($haystack, $needle) {
    return strpos($haystack, $needle) === 0;
}

function php7_str_ends_with($haystack, $needle) {
    $count = strlen($needle);
    if ($count === 0) {
        return true;
    }
    return substr($haystack, -$count) === $needle;
}

// Location class
class Loc {
    public $file_path;
    public $row;
    public $col;

    public function __construct($file_path, $row, $col) {
        $this->file_path = $file_path;
        $this->row = $row;
        $this->col = $col;
    }

    public function display() {
        return sprintf("%s:%d:%d", $this->file_path, $this->row + 1, $this->col + 1);
    }
}

// Token types
define("TOKEN_NAME", "TOKEN_NAME");
define("TOKEN_OPAREN", "TOKEN_OPAREN");
define("TOKEN_CPAREN", "TOKEN_CPAREN");
define("TOKEN_OCURLY", "TOKEN_OCURLY");
define("TOKEN_CCURLY", "TOKEN_CCURLY");
define("TOKEN_COMMA", "TOKEN_COMMA");
define("TOKEN_SEMICOLON", "TOKEN_SEMICOLON");
define("TOKEN_NUMBER", "TOKEN_NUMBER");
define("TOKEN_STRING", "TOKEN_STRING");
define("TOKEN_RETURN", "TOKEN_RETURN");
define("TOKEN_CLASS", "TOKEN_CLASS");
define("TOKEN_MODULE", "TOKEN_MODULE");
define("TOKEN_NAMESPACE", "TOKEN_NAMESPACE");
define("TOKEN_IF", "TOKEN_IF");
define("TOKEN_ELSE", "TOKEN_ELSE");
define("TOKEN_WHILE", "TOKEN_WHILE");
define("TOKEN_FOR", "TOKEN_FOR");
define("TOKEN_SWITCH", "TOKEN_SWITCH");
define("TOKEN_CASE", "TOKEN_CASE");
define("TOKEN_BREAK", "TOKEN_BREAK");
define("TOKEN_CONTINUE", "TOKEN_CONTINUE");

// Token class
class Token {
    public $type;
    public $value;
    public $loc;

    public function __construct($loc, $type, $value) {
        $this->loc = $loc;
        $this->type = $type;
        $this->value = $value;
    }
}

// Lexer class
class Lexer {
    public $file_path;
    public $source;
    public $cur;
    public $bol;
    public $row;

    public function __construct($file_path, $source) {
        $this->file_path = $file_path;
        $this->source = $source;
        $this->cur = 0;
        $this->bol = 0;
        $this->row = 0;
    }

    function is_not_empty() {
        return $this->cur < strlen($this->source);
    }

    function is_empty() {
        return !$this->is_not_empty();
    }

    function chop_char() {
        if ($this->is_not_empty()) {
            $x = $this->source[$this->cur];
            $this->cur += 1;
            if ($x === "\n") {
                $this->bol = $this->cur;
                $this->row += 1;
            }
        }
    }

    function loc() {
        return new Loc($this->file_path, $this->row, $this->cur - $this->bol);
    }

    function trim_left() {
        while ($this->is_not_empty() && ctype_space($this->source[$this->cur])) {
            $this->chop_char();
        }
    }

    function drop_line() {
        while ($this->is_not_empty() && $this->source[$this->cur] !== "\n") {
            $this->chop_char();
        }
        if ($this->is_not_empty()) {
            $this->chop_char();
        }
    }

    function next_token() {
        $this->trim_left();
        while ($this->is_not_empty()) {
            $s = substr($this->source, $this->cur);
            if (!php7_str_starts_with($s, "#") && !php7_str_starts_with($s, "//")) break;
            $this->drop_line();
            $this->trim_left();
        }
        if ($this->is_empty()) {
            return false;
        }
        $loc = $this->loc();
        $first = $this->source[$this->cur];
        if (ctype_alpha($first)) {
            $index = $this->cur;
            while ($this->is_not_empty() && ctype_alnum($this->source[$this->cur])) {
                $this->chop_char();
            }
            $value = substr($this->source, $index, $this->cur - $index);
            $keywords = array(
                "class" => TOKEN_CLASS,
                "module" => TOKEN_MODULE,
                "namespace" => TOKEN_NAMESPACE,
                "if" => TOKEN_IF,
                "else" => TOKEN_ELSE,
                "while" => TOKEN_WHILE,
                "for" => TOKEN_FOR,
                "switch" => TOKEN_SWITCH,
                "case" => TOKEN_CASE,
                "break" => TOKEN_BREAK,
                "continue" => TOKEN_CONTINUE,
                "return" => TOKEN_RETURN,
            );
            if (isset($keywords[$value])) {
                return new Token($loc, $keywords[$value], $value);
            }
            return new Token($loc, TOKEN_NAME, $value);
        } elseif ($first === '"') {
            $this->chop_char();
            $start = $this->cur;
            $literal = "";
            while ($this->is_not_empty()) {
                $ch = $this->source[$this->cur];
                switch ($ch) {
                    case '"':
                        break 2;
                    case '\\':
                        $this->chop_char();
                        if ($this->is_empty()) {
                            print("{$this->loc()->display()}: ERROR: unfinished escape sequence\n");
                            exit(69);
                        }
                        $escape = $this->source[$this->cur];
                        switch ($escape) {
                            case 'n':
                                $literal .= "\n";
                                $this->chop_char();
                                break;
                            case '"':
                                $literal .= "\"";
                                $this->chop_char();
                                break;
                            default:
                                print("{$this->loc()->display()}: ERROR: unknown escape sequence starts with {$escape}\n");
                        }
                        break;
                    default:
                        $literal .= $ch;
                        $this->chop_char();
                }
            }
            if ($this->is_not_empty()) {
                $this->chop_char();
                return new Token($loc, TOKEN_STRING, $literal);
            }
            echo sprintf("%s: ERROR: unclosed string literal\n", $loc->display());
            exit(69);
        } elseif (ctype_digit($first)) {
            $start = $this->cur;
            while ($this->is_not_empty() && ctype_digit($this->source[$this->cur])) {
                $this->chop_char();
            }
            $value = (int)substr($this->source, $start, $this->cur - $start);
            return new Token($loc, TOKEN_NUMBER, $value);
        } elseif ($first === '\'') {
            // Character literal
            $this->chop_char();
            if ($this->is_empty()) {
                print("{$this->loc()->display()}: ERROR: unfinished character literal\n");
                exit(69);
            }
            $ch = $this->source[$this->cur];
            $this->chop_char();
            if ($this->is_empty() || $this->source[$this->cur] !== '\'') {
                print("{$this->loc()->display()}: ERROR: unclosed character literal\n");
                exit(69);
            }
            $this->chop_char();
            return new Token($loc, TOKEN_NUMBER, ord($ch));
        } elseif (in_array($first, array('(', ')', '{', '}', ',', ';'))) {
            $token_type = null;
            switch ($first) {
                case '(':
                    $token_type = TOKEN_OPAREN;
                    break;
                case ')':
                    $token_type = TOKEN_CPAREN;
                    break;
                case '{':
                    $token_type = TOKEN_OCURLY;
                    break;
                case '}':
                    $token_type = TOKEN_CCURLY;
                    break;
                case ',':
                    $token_type = TOKEN_COMMA;
                    break;
                case ';':
                    $token_type = TOKEN_SEMICOLON;
                    break;
            }
            $this->chop_char();
            return new Token($loc, $token_type, $first);
        } else {
            print("{$this->loc()->display()}: ERROR: unknown token starts with {$first}\n");
            exit(69);
        }
    }
}

// Type definitions
define("TYPE_INT", "TYPE_INT");
define("TYPE_STRING", "TYPE_STRING");

// AST nodes
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

class VarDecl {
    public $name;
    public $type;
    public $init;

    public function __construct($name, $type, $init) {
        $this->name = $name;
        $this->type = $type;
        $this->init = $init;
    }
}

// Expression nodes
class NumberExpr {
    public $value;

    public function __construct($value) {
        $this->value = $value;
    }
}

class StringExpr {
    public $value;

    public function __construct($value) {
        $this->value = $value;
    }
}

class VarExpr {
    public $name;

    public function __construct($name) {
        $this->name = $name;
    }
}

// Parser class
class Parser {
    public $lexer;
    public $cur_token;

    public function __construct($lexer) {
        $this->lexer = $lexer;
        $this->next_token();
    }

    function next_token() {
        $this->cur_token = $this->lexer->next_token();
    }

    function parse_stmt() {
        if ($this->cur_token->type === TOKEN_RETURN) {
            $this->next_token();
            $expr = $this->parse_expr();
            return new RetStmt($expr);
        } elseif ($this->cur_token->type === TOKEN_NAME) {
            $name = $this->cur_token->value;
            $this->next_token();
            if ($this->cur_token->type === TOKEN_OPAREN) {
                $this->next_token();
                $args = [];
                while ($this->cur_token->type !== TOKEN_CPAREN) {
                    $args[] = $this->parse_expr();
                    if ($this->cur_token->type === TOKEN_COMMA) {
                        $this->next_token();
                    }
                }
                $this->next_token(); // Consume the closing parenthesis
                return new FuncallStmt($name, $args);
            }
        }
        print("ERROR: Unexpected token in statement\n");
        exit(69);
    }

    function parse_expr() {
        if ($this->cur_token->type === TOKEN_NUMBER) {
            $value = $this->cur_token->value;
            $this->next_token();
            return new NumberExpr($value);
        } elseif ($this->cur_token->type === TOKEN_STRING) {
            $value = $this->cur_token->value;
            $this->next_token();
            return new StringExpr($value);
        } elseif ($this->cur_token->type === TOKEN_NAME) {
            $name = $this->cur_token->value;
            $this->next_token();
            return new VarExpr($name);
        }
        print("ERROR: Unexpected token in expression\n");
        exit(69);
    }
}

// Main program logic
function main($argv) {
    if (count($argv) < 2) {
        print("Usage: php parser.php <file>\n");
        exit(1);
    }

    $file_path = $argv[1];
    if (!file_exists($file_path)) {
        print("ERROR: File not found: $file_path\n");
        exit(1);
    }

    $source = file_get_contents($file_path);
    $lexer = new Lexer($file_path, $source);
    $parser = new Parser($lexer);

    while ($lexer->is_not_empty()) {
        $stmt = $parser->parse_stmt();
        if ($stmt instanceof FuncallStmt) {
            printf("Function call: %s(%s)\n", $stmt->name, implode(", ", array_map(function($arg) {
                return $arg instanceof VarExpr ? $arg->name : (string) $arg->value;
            }, $stmt->args)));
        } elseif ($stmt instanceof RetStmt) {
            printf("Return statement: %s\n", $stmt->expr->value);
        }
    }
}

main($argv);


