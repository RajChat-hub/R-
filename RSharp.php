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
    public $value;

    public function __construct($name, $type, $value) {
        $this->name = $name;
        $this->type = $type;
        $this->value = $value;
    }
}

class IfStmt {
    public $condition;
    public $then_branch;
    public $else_branch;

    public function __construct($condition, $then_branch, $else_branch) {
        $this->condition = $condition;
        $this->then_branch = $then_branch;
        $this->else_branch = $else_branch;
    }
}

class WhileStmt {
    public $condition;
    public $body;

    public function __construct($condition, $body) {
        $this->condition = $condition;
        $this->body = $body;
    }
}

class ForStmt {
    public $init;
    public $condition;
    public $increment;
    public $body;

    public function __construct($init, $condition, $increment, $body) {
        $this->init = $init;
        $this->condition = $condition;
        $this->increment = $increment;
        $this->body = $body;
    }
}

class Program {
    public $statements;

    public function __construct($statements) {
        $this->statements = $statements;
    }
}

// Parser class
class Parser {
    private $lexer;
    private $current_token;

    public function __construct($lexer) {
        $this->lexer = $lexer;
        $this->current_token = $this->lexer->next_token();
    }

    private function error() {
        echo "Syntax error\n";
        exit(1);
    }

    private function consume($token_type) {
        if ($this->current_token->type === $token_type) {
            $this->current_token = $this->lexer->next_token();
        } else {
            $this->error();
        }
    }

    public function parse() {
        $statements = [];
        while ($this->current_token !== false) {
            $statements[] = $this->statement();
        }
        return new Program($statements);
    }

    private function statement() {
        switch ($this->current_token->type) {
            case TOKEN_RETURN:
                return $this->return_stmt();
            case TOKEN_IF:
                return $this->if_stmt();
            case TOKEN_WHILE:
                return $this->while_stmt();
            case TOKEN_FOR:
                return $this->for_stmt();
            case TOKEN_CLASS:
            case TOKEN_MODULE:
            case TOKEN_NAMESPACE:
                return $this->var_decl();
            default:
                return $this->funcall_stmt();
        }
    }

    private function return_stmt() {
        $this->consume(TOKEN_RETURN);
        $expr = $this->expression();
        $this->consume(TOKEN_SEMICOLON);
        return new RetStmt($expr);
    }

    private function if_stmt() {
        $this->consume(TOKEN_IF);
        $this->consume(TOKEN_OPAREN);
        $condition = $this->expression();
        $this->consume(TOKEN_CPAREN);
        $this->consume(TOKEN_OCURLY);
        $then_branch = $this->block();
        $this->consume(TOKEN_CCURLY);
        $else_branch = null;
        if ($this->current_token->type === TOKEN_ELSE) {
            $this->consume(TOKEN_ELSE);
            $this->consume(TOKEN_OCURLY);
            $else_branch = $this->block();
            $this->consume(TOKEN_CCURLY);
        }
        return new IfStmt($condition, $then_branch, $else_branch);
    }

    private function while_stmt() {
        $this->consume(TOKEN_WHILE);
        $this->consume(TOKEN_OPAREN);
        $condition = $this->expression();
        $this->consume(TOKEN_CPAREN);
        $this->consume(TOKEN_OCURLY);
        $body = $this->block();
        $this->consume(TOKEN_CCURLY);
        return new WhileStmt($condition, $body);
    }

    private function for_stmt() {
        $this->consume(TOKEN_FOR);
        $this->consume(TOKEN_OPAREN);
        $init = $this->var_decl();
        $this->consume(TOKEN_SEMICOLON);
        $condition = $this->expression();
        $this->consume(TOKEN_SEMICOLON);
        $increment = $this->funcall_stmt();
        $this->consume(TOKEN_CPAREN);
        $this->consume(TOKEN_OCURLY);
        $body = $this->block();
        $this->consume(TOKEN_CCURLY);
        return new ForStmt($init, $condition, $increment, $body);
    }

    private function block() {
        $statements = [];
        while ($this->current_token->type !== TOKEN_CCURLY) {
            $statements[] = $this->statement();
        }
        return $statements;
    }

    private function funcall_stmt() {
        $name = $this->current_token->value;
        $this->consume(TOKEN_NAME);
        $this->consume(TOKEN_OPAREN);
        $args = [];
        while ($this->current_token->type !== TOKEN_CPAREN) {
            $args[] = $this->expression();
            if ($this->current_token->type === TOKEN_COMMA) {
                $this->consume(TOKEN_COMMA);
            }
        }
        $this->consume(TOKEN_CPAREN);
        $this->consume(TOKEN_SEMICOLON);
        return new FuncallStmt($name, $args);
    }

    private function var_decl() {
        $name = $this->current_token->value;
        $this->consume(TOKEN_NAME);
        $this->consume(TOKEN_OPAREN);
        $type = $this->current_token->value;
        $this->consume(TOKEN_NAME);
        $this->consume(TOKEN_CPAREN);
        $value = $this->expression();
        $this->consume(TOKEN_SEMICOLON);
        return new VarDecl($name, $type, $value);
    }

    private function expression() {
        // Placeholder for expression parsing
        $expr = $this->current_token->value;
        $this->consume($this->current_token->type);
        return $expr;
    }
}

// Main execution
if ($argc < 2) {
    echo "Usage: php parser.php <file>\n";
    exit(1);
}

$file_path = $argv[1];
$source = file_get_contents($file_path);
$lexer = new Lexer($file_path, $source);
$parser = new Parser($lexer);
$program = $parser->parse();

echo "Parsed program successfully.\n";

?>