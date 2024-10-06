// math.js

// --- Basic Arithmetic ---
export function add(a, b) {
  if (typeof a !== 'number' || typeof b !== 'number') throw new Error('Inputs must be numbers');
  return a + b;
}

export function subtract(a, b) {
  if (typeof a !== 'number' || typeof b !== 'number') throw new Error('Inputs must be numbers');
  return a - b;
}

export function multiply(a, b) {
  if (typeof a !== 'number' || typeof b !== 'number') throw new Error('Inputs must be numbers');
  return a * b;
}

export function divide(a, b) {
  if (typeof a !== 'number' || typeof b !== 'number') throw new Error('Inputs must be numbers');
  if (b === 0) throw new Error('Division by zero');
  return a / b;
}

// --- Exponents and Roots ---
export function power(base, exponent) {
  if (typeof base !== 'number' || typeof exponent !== 'number') throw new Error('Inputs must be numbers');
  return Math.pow(base, exponent);
}

export function sqrt(value) {
  if (typeof value !== 'number') throw new Error('Input must be a number');
  if (value < 0) throw new Error('Cannot compute the square root of a negative number');
  return Math.sqrt(value);
}

// --- Trigonometry ---
export function sin(angle) {
  return Math.sin(toRadians(angle));
}

export function cos(angle) {
  return Math.cos(toRadians(angle));
}

export function tan(angle) {
  return Math.tan(toRadians(angle));
}

export function toRadians(degrees) {
  return degrees * (Math.PI / 180);
}

export function toDegrees(radians) {
  return radians * (180 / Math.PI);
}

// --- Logarithms and Exponentials ---
export function log(value, base = Math.E) {
  if (typeof value !== 'number' || value <= 0) throw new Error('Value must be a positive number');
  return Math.log(value) / Math.log(base);
}

export function exp(value) {
  if (typeof value !== 'number') throw new Error('Input must be a number');
  return Math.exp(value);
}

// --- Factorials ---
export function factorial(n) {
  if (n < 0) throw new Error('Negative numbers do not have factorials');
  if (n === 0 || n === 1) return 1;
  let result = 1;
  for (let i = 2; i <= n; i++) {
    result *= i;
  }
  return result;
}

// --- GCD and LCM ---
export function gcd(a, b) {
  if (typeof a !== 'number' || typeof b !== 'number') throw new Error('Inputs must be numbers');
  while (b !== 0) {
    [a, b] = [b, a % b];
  }
  return Math.abs(a);
}

export function lcm(a, b) {
  if (typeof a !== 'number' || typeof b !== 'number') throw new Error('Inputs must be numbers');
  return Math.abs(a * b) / gcd(a, b);
}

// --- Statistics ---
export function mean(values) {
  if (!Array.isArray(values)) throw new Error('Input must be an array');
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function median(values) {
  if (!Array.isArray(values)) throw new Error('Input must be an array');
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function mode(values) {
  if (!Array.isArray(values)) throw new Error('Input must be an array');
  const frequency = {};
  values.forEach(v => frequency[v] = (frequency[v] || 0) + 1);
  const maxFrequency = Math.max(...Object.values(frequency));
  return Object.keys(frequency).filter(v => frequency[v] === maxFrequency);
}

export function variance(values) {
  const m = mean(values);
  return mean(values.map(v => (v - m) ** 2));
}

export function standardDeviation(values) {
  return Math.sqrt(variance(values));
}

// --- Random Numbers ---
export function randomInt(min, max) {
  if (typeof min !== 'number' || typeof max !== 'number') throw new Error('Inputs must be numbers');
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function randomFloat(min, max) {
  if (typeof min !== 'number' || typeof max !== 'number') throw new Error('Inputs must be numbers');
  return Math.random() * (max - min) + min;
}

// --- Matrix Operations ---
export function createMatrix(rows, cols, fillValue = 0) {
  return Array.from({ length: rows }, () => Array(cols).fill(fillValue));
}

export function transpose(matrix) {
  if (!Array.isArray(matrix) || !Array.isArray(matrix[0])) throw new Error('Input must be a 2D array');
  return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

export function multiplyMatrices(A, B) {
  if (!Array.isArray(A) || !Array.isArray(B) || !Array.isArray(A[0]) || !Array.isArray(B[0])) throw new Error('Inputs must be 2D arrays');
  if (A[0].length !== B.length) throw new Error('Matrix dimensions must be compatible for multiplication');
  
  const result = createMatrix(A.length, B[0].length);
  
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < B[0].length; j++) {
      for (let k = 0; k < B.length; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
}

// --- Complex Numbers ---
export function addComplex(a, b) {
  return { real: a.real + b.real, imag: a.imag + b.imag };
}

export function multiplyComplex(a, b) {
  return {
    real: a.real * b.real - a.imag * b.imag,
    imag: a.real * b.imag + a.imag * b.real
  };
}

// --- Formatting ---
export function toFixed(value, decimals) {
  if (typeof value !== 'number') throw new Error('Input must be a number');
  return value.toFixed(decimals);
}

export function toPrecision(value, precision) {
  if (typeof value !== 'number') throw new Error('Input must be a number');
  return value.toPrecision(precision);
}

// --- Prime Numbers ---
export function isPrime(num) {
  if (typeof num !== 'number' || num < 2) return false;
  for (let i = 2; i <= Math.sqrt(num); i++) {
    if (num % i === 0) return false;
  }
  return true;
}

export function generatePrimes(limit) {
  const primes = [];
  for (let i = 2; i <= limit; i++) {
    if (isPrime(i)) primes.push(i);
  }
  return primes;
}

// --- Fibonacci ---
export function fibonacci(n) {
  if (n <= 1) return n;
  let a = 0, b = 1, temp;
  for (let i = 2; i <= n; i++) {
    temp = a + b;
    a = b;
    b = temp;
  }
  return b;
}

// --- Conversions ---
export function degreesToRadians(degrees) {
  return degrees * (Math.PI / 180);
}

export function radiansToDegrees(radians) {
  return radians * (180 / Math.PI);
}

// --- Geometry ---
export function areaOfCircle(radius) {
  if (typeof radius !== 'number' || radius <= 0) throw new Error('Radius must be a positive number');
  return Math.PI * radius * radius;
}

export function circumferenceOfCircle(radius) {
  if (typeof radius !== 'number' || radius <= 0) throw new Error('Radius must be a positive number');
  return 2 * Math.PI * radius;
}

// More advanced or additional functions can be added here as needed for your project.

// --- Additional Functions ---

// --- Advanced Geometry ---
export function areaOfRectangle(length, width) {
  if (length <= 0 || width <= 0) throw new Error('Length and width must be positive numbers');
  return length * width;
}

export function perimeterOfRectangle(length, width) {
  if (length <= 0 || width <= 0) throw new Error('Length and width must be positive numbers');
  return 2 * (length + width);
}

export function areaOfTriangle(base, height) {
  if (base <= 0 || height <= 0) throw new Error('Base and height must be positive numbers');
  return 0.5 * base * height;
}

export function perimeterOfTriangle(a, b, c) {
  if (a <= 0 || b <= 0 || c <= 0) throw new Error('All sides must be positive numbers');
  return a + b + c;
}

export function areaOfTrapezoid(base1, base2, height) {
  if (base1 <= 0 || base2 <= 0 || height <= 0) throw new Error('Base and height must be positive numbers');
  return ((base1 + base2) / 2) * height;
}

export function volumeOfCylinder(radius, height) {
  if (radius <= 0 || height <= 0) throw new Error('Radius and height must be positive numbers');
  return Math.PI * Math.pow(radius, 2) * height;
}

export function surfaceAreaOfCylinder(radius, height) {
  if (radius <= 0 || height <= 0) throw new Error('Radius and height must be positive numbers');
  return 2 * Math.PI * radius * (radius + height);
}

// --- Advanced Trigonometry ---
export function sec(angle) {
  return 1 / Math.cos(toRadians(angle));
}

export function csc(angle) {
  return 1 / Math.sin(toRadians(angle));
}

export function cot(angle) {
  return 1 / Math.tan(toRadians(angle));
}

export function arcSin(value) {
  if (value < -1 || value > 1) throw new Error('Input must be between -1 and 1');
  return Math.asin(value);
}

export function arcCos(value) {
  if (value < -1 || value > 1) throw new Error('Input must be between -1 and 1');
  return Math.acos(value);
}

export function arcTan(value) {
  return Math.atan(value);
}

// --- Advanced Probability and Combinatorics ---
export function combinations(n, r) {
  if (n < r) throw new Error('n must be greater than or equal to r');
  return factorial(n) / (factorial(r) * factorial(n - r));
}

export function permutations(n, r) {
  if (n < r) throw new Error('n must be greater than or equal to r');
  return factorial(n) / factorial(n - r);
}

export function probability(successes, trials) {
  if (successes > trials || trials <= 0) throw new Error('Invalid number of successes or trials');
  return successes / trials;
}

export function binomialProbability(successes, trials, probabilityOfSuccess) {
  return combinations(trials, successes) * Math.pow(probabilityOfSuccess, successes) * Math.pow(1 - probabilityOfSuccess, trials - successes);
}

// --- Higher Mathematics: Calculus ---
export function derivative(f, h = 1e-5) {
  return function (x) {
    return (f(x + h) - f(x)) / h;
  };
}

export function definiteIntegral(f, a, b, n = 1000) {
  const width = (b - a) / n;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += f(a + i * width) * width;
  }
  return sum;
}

// --- Vector Operations ---
export function addVectors(v1, v2) {
  if (v1.length !== v2.length) throw new Error('Vectors must have the same dimensions');
  return v1.map((val, index) => val + v2[index]);
}

export function dotProduct(v1, v2) {
  if (v1.length !== v2.length) throw new Error('Vectors must have the same dimensions');
  return v1.reduce((sum, val, index) => sum + val * v2[index], 0);
}

export function crossProduct(v1, v2) {
  if (v1.length !== 3 || v2.length !== 3) throw new Error('Cross product is only defined for 3D vectors');
  return [
    v1[1] * v2[2] - v1[2] * v2[1],
    v1[2] * v2[0] - v1[0] * v2[2],
    v1[0] * v2[1] - v1[1] * v2[0],
  ];
}

// --- Matrix Operations ---
export function inverseMatrix(matrix) {
  const determinant = determinantMatrix(matrix);
  if (determinant === 0) throw new Error('Matrix is not invertible');
  const adjugate = adjugateMatrix(matrix);
  return adjugate.map(row => row.map(value => value / determinant));
}

export function determinantMatrix(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== matrix[0].length) throw new Error('Input must be a square matrix');
  
  if (matrix.length === 2) {
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
  }
  
  let determinant = 0;
  for (let i = 0; i < matrix[0].length; i++) {
    determinant += (i % 2 === 0 ? 1 : -1) * matrix[0][i] * determinantMatrix(minorMatrix(matrix, 0, i));
  }
  return determinant;
}

export function minorMatrix(matrix, row, col) {
  return matrix
    .filter((_, rowIndex) => rowIndex !== row)
    .map(row => row.filter((_, colIndex) => colIndex !== col));
}

export function adjugateMatrix(matrix) {
  return matrix.map((row, i) =>
    row.map((_, j) => (i + j) % 2 === 0 ? determinantMatrix(minorMatrix(matrix, i, j)) : -determinantMatrix(minorMatrix(matrix, i, j)))
  );
}

// --- Number Theory ---
export function isEven(number) {
  return number % 2 === 0;
}

export function isOdd(number) {
  return number % 2 !== 0;
}

export function modularExponentiation(base, exponent, mod) {
  if (mod <= 0) throw new Error('Modulus must be a positive integer');
  let result = 1;
  base = base % mod;
  while (exponent > 0) {
    if (exponent % 2 === 1) result = (result * base) % mod;
    exponent = Math.floor(exponent / 2);
    base = (base * base) % mod;
  }
  return result;
}

export function greatestCommonDivisor(a, b) {
  while (b !== 0) {
    [a, b] = [b, a % b];
  }
  return a;
}

export function leastCommonMultiple(a, b) {
  return Math.abs(a * b) / greatestCommonDivisor(a, b);
}

// --- Advanced Number Formatting ---
export function formatScientific(value, digits = 4) {
  if (typeof value !== 'number') throw new Error('Input must be a number');
  return value.toExponential(digits);
}

export function formatCurrency(value, locale = 'en-US', currency = 'USD') {
  return new Intl.NumberFormat(locale, { style: 'currency', currency }).format(value);
}

// --- Probability Distributions ---
export function normalDistribution(x, mean = 0, stddev = 1) {
  const exponent = -0.5 * Math.pow((x - mean) / stddev, 2);
  return (1 / (stddev * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
}

export function poissonDistribution(lambda, k) {
  return (Math.pow(lambda, k) * Math.exp(-lambda)) / factorial(k);
}

// math.js - Complex Number Support

// Complex number class definition
class Complex {
    constructor(real, imaginary) {
        this.real = real;        // Real part
        this.imaginary = imaginary; // Imaginary part
    }
}

// Addition of two complex numbers
function add(complex1, complex2) {
    return new Complex(complex1.real + complex2.real, complex1.imaginary + complex2.imaginary);
}

// Subtraction of two complex numbers
function subtract(complex1, complex2) {
    return new Complex(complex1.real - complex2.real, complex1.imaginary - complex2.imaginary);
}

// Multiplication of two complex numbers
function multiply(complex1, complex2) {
    const realPart = complex1.real * complex2.real - complex1.imaginary * complex2.imaginary;
    const imaginaryPart = complex1.real * complex2.imaginary + complex1.imaginary * complex2.real;
    return new Complex(realPart, imaginaryPart);
}

// Division of two complex numbers
function divide(complex1, complex2) {
    const denominator = complex2.real ** 2 + complex2.imaginary ** 2;
    const realPart = (complex1.real * complex2.real + complex1.imaginary * complex2.imaginary) / denominator;
    const imaginaryPart = (complex1.imaginary * complex2.real - complex1.real * complex2.imaginary) / denominator;
    return new Complex(realPart, imaginaryPart);
}

// Magnitude (Modulus) of a complex number
function magnitude(complex) {
    return Math.sqrt(complex.real ** 2 + complex.imaginary ** 2);
}

// Conjugate of a complex number
function conjugate(complex) {
    return new Complex(complex.real, -complex.imaginary);
}

// Argument (Phase) of a complex number
function argument(complex) {
    return Math.atan2(complex.imaginary, complex.real);
}

// String representation of a complex number
function toString(complex) {
    return `${complex.real} + ${complex.imaginary}i`;
}

// Example Usage
const complex1 = new Complex(3, 2);
const complex2 = new Complex(1, 7);

console.log(`Complex Number 1: ${toString(complex1)}`);
console.log(`Complex Number 2: ${toString(complex2)}`);

// Performing operations
const sum = add(complex1, complex2);
console.log(`Sum: ${toString(sum)}`);

const difference = subtract(complex1, complex2);
console.log(`Difference: ${toString(difference)}`);

const product = multiply(complex1, complex2);
console.log(`Product: ${toString(product)}`);

const quotient = divide(complex1, complex2);
console.log(`Quotient: ${toString(quotient)}`);

const magnitude1 = magnitude(complex1);
console.log(`Magnitude of complex1: ${magnitude1}`);

const conjugate1 = conjugate(complex1);
console.log(`Conjugate of complex1: ${toString(conjugate1)}`);

const argument1 = argument(complex1);
console.log(`Argument of complex1: ${argument1}`);

// Array of Complex Numbers
function addComplexArray(complexArray) {
    return complexArray.reduce((acc, complex) => add(acc, complex), new Complex(0, 0));
}

function multiplyComplexArray(complexArray) {
    return complexArray.reduce((acc, complex) => multiply(acc, complex), new Complex(1, 0));
}

// Testing Array Operations
const complexArray = [new Complex(1, 2), new Complex(3, 4), new Complex(5, -1)];
const arraySum = addComplexArray(complexArray);
console.log(`Sum of Complex Array: ${toString(arraySum)}`);

const arrayProduct = multiplyComplexArray(complexArray);
console.log(`Product of Complex Array: ${toString(arrayProduct)}`);

// Error Handling
function safeDivide(complex1, complex2) {
    const denominator = complex2.real ** 2 + complex2.imaginary ** 2;
    if (denominator === 0) {
        throw new Error('Division by zero error: The denominator is zero.');
    }
    return divide(complex1, complex2);
}

// Testing safeDivide
try {
    const complexZero = new Complex(0, 0);
    const result = safeDivide(complex1, complexZero);
    console.log(`Safe Division Result: ${toString(result)}`);
} catch (error) {
    console.error(error.message);
}

// Extended Features

// Power of a Complex Number
function power(complex, exponent) {
    const magnitude = Math.pow(magnitude(complex), exponent);
    const angle = argument(complex) * exponent;
    return new Complex(magnitude * Math.cos(angle), magnitude * Math.sin(angle));
}

// Testing Power Function
const complexPower = power(complex1, 2);
console.log(`Complex1 squared: ${toString(complexPower)}`);

// Exponential Function
function exponential(complex) {
    const expReal = Math.exp(complex.real);
    return new Complex(expReal * Math.cos(complex.imaginary), expReal * Math.sin(complex.imaginary));
}

// Testing Exponential Function
const complexExponential = exponential(complex1);
console.log(`Exponential of Complex1: ${toString(complexExponential)}`);

// Logarithm of a Complex Number
function logarithm(complex) {
    const realPart = Math.log(magnitude(complex));
    const imaginaryPart = argument(complex);
    return new Complex(realPart, imaginaryPart);
}

// Testing Logarithm Function
const complexLogarithm = logarithm(complex1);
console.log(`Logarithm of Complex1: ${toString(complexLogarithm)}`);

// math.js - Advanced Matrix Operations

class Matrix {
    constructor(data) {
        this.data = data; // 2D array representing the matrix
        this.rows = data.length; // Number of rows
        this.cols = data[0].length; // Number of columns
    }

    // Get the value at a specific row and column
    get(row, col) {
        return this.data[row][col];
    }

    // Set the value at a specific row and column
    set(row, col, value) {
        this.data[row][col] = value;
    }

    // Print the matrix
    print() {
        this.data.forEach(row => console.log(row));
    }

    // Matrix addition
    static add(A, B) {
        if (A.rows !== B.rows || A.cols !== B.cols) {
            throw new Error('Matrix dimensions must match for addition.');
        }
        const result = new Array(A.rows).fill(0).map(() => new Array(A.cols).fill(0));
        for (let i = 0; i < A.rows; i++) {
            for (let j = 0; j < A.cols; j++) {
                result[i][j] = A.get(i, j) + B.get(i, j);
            }
        }
        return new Matrix(result);
    }

    // Matrix subtraction
    static subtract(A, B) {
        if (A.rows !== B.rows || A.cols !== B.cols) {
            throw new Error('Matrix dimensions must match for subtraction.');
        }
        const result = new Array(A.rows).fill(0).map(() => new Array(A.cols).fill(0));
        for (let i = 0; i < A.rows; i++) {
            for (let j = 0; j < A.cols; j++) {
                result[i][j] = A.get(i, j) - B.get(i, j);
            }
        }
        return new Matrix(result);
    }

    // Matrix multiplication
    static multiply(A, B) {
        if (A.cols !== B.rows) {
            throw new Error('Matrix A columns must match Matrix B rows for multiplication.');
        }
        const result = new Array(A.rows).fill(0).map(() => new Array(B.cols).fill(0));
        for (let i = 0; i < A.rows; i++) {
            for (let j = 0; j < B.cols; j++) {
                for (let k = 0; k < A.cols; k++) {
                    result[i][j] += A.get(i, k) * B.get(k, j);
                }
            }
        }
        return new Matrix(result);
    }

    // Matrix transpose
    transpose() {
        const result = new Array(this.cols).fill(0).map(() => new Array(this.rows).fill(0));
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result[j][i] = this.get(i, j);
            }
        }
        return new Matrix(result);
    }

    // Calculate the determinant of a matrix
    determinant() {
        if (this.rows !== this.cols) {
            throw new Error('Determinant can only be calculated for square matrices.');
        }
        if (this.rows === 2) {
            return this.get(0, 0) * this.get(1, 1) - this.get(0, 1) * this.get(1, 0);
        }
        let det = 0;
        for (let i = 0; i < this.cols; i++) {
            det += this.get(0, i) * this.cofactor(0, i);
        }
        return det;
    }

    // Calculate the cofactor of a matrix element
    cofactor(row, col) {
        const minor = this.minor(row, col);
        return ((row + col) % 2 === 0 ? 1 : -1) * minor.determinant();
    }

    // Calculate the minor of a matrix element
    minor(row, col) {
        const subMatrix = this.data
            .map((r, i) => r.filter((_, j) => i !== row && j !== col));
        return new Matrix(subMatrix);
    }

    // Calculate the inverse of a matrix
    inverse() {
        if (this.rows !== this.cols) {
            throw new Error('Inverse can only be calculated for square matrices.');
        }
        const det = this.determinant();
        if (det === 0) {
            throw new Error('Matrix is singular and cannot be inverted.');
        }
        const result = new Array(this.rows).fill(0).map(() => new Array(this.cols).fill(0));
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result[j][i] = this.cofactor(i, j) / det; // Note the transposition
            }
        }
        return new Matrix(result);
    }

    // Calculate eigenvalues using the characteristic polynomial
    eigenvalues() {
        const A = this.data;
        const n = this.rows;
        const charPoly = this.characteristicPolynomial();
        return this.solvePolynomial(charPoly);
    }

    // Calculate the characteristic polynomial
    characteristicPolynomial() {
        const A = this.data;
        const n = this.rows;
        const coeffs = new Array(n + 1).fill(0);
        for (let i = 0; i < n; i++) {
            const subMatrix = this.minor(i, i);
            coeffs[i] = (i % 2 === 0 ? 1 : -1) * A[i][i] * subMatrix.determinant();
        }
        coeffs[n] = -1; // x^n term
        return coeffs;
    }

    // Solve polynomial (e.g. using Newton's method)
    solvePolynomial(coeffs) {
        // Placeholder for root-finding logic (e.g. Newton-Raphson)
        return []; // Implement polynomial solving here
    }

    // Calculate eigenvectors (using the Jordan form, for example)
    eigenvectors() {
        const values = this.eigenvalues();
        const vectors = values.map(value => this.eigenvector(value));
        return vectors;
    }

    // Calculate the eigenvector corresponding to an eigenvalue
    eigenvector(eigenvalue) {
        // Placeholder for eigenvector calculation
        return []; // Implement eigenvector logic here
    }
}

// Example Usage
const matrixA = new Matrix([[4, 2], [3, 1]]);
const matrixB = new Matrix([[1, 0], [0, 1]]);

// Matrix addition
const sum = Matrix.add(matrixA, matrixB);
console.log('Sum:');
sum.print();

// Matrix subtraction
const difference = Matrix.subtract(matrixA, matrixB);
console.log('Difference:');
difference.print();

// Matrix multiplication
const product = Matrix.multiply(matrixA, matrixB);
console.log('Product:');
product.print();

// Matrix inverse
const inverseA = matrixA.inverse();
console.log('Inverse of A:');
inverseA.print();

// Matrix determinant
const detA = matrixA.determinant();
console.log(`Determinant of A: ${detA}`);

// Eigenvalues and eigenvectors
const eigenvalues = matrixA.eigenvalues();
console.log(`Eigenvalues: ${eigenvalues}`);

const eigenvectors = matrixA.eigenvectors();
console.log(`Eigenvectors:`);
eigenvectors.forEach((vec, index) => {
    console.log(`Eigenvector ${index + 1}: ${vec}`);
});

// Matrix transpose
const transposed = matrixA.transpose();
console.log('Transposed Matrix A:');
transposed.print();

// math.js - Statistical Functions

class Statistics {
    // Calculate mean
    static mean(data) {
        if (data.length === 0) {
            throw new Error("Data array cannot be empty.");
        }
        const sum = data.reduce((acc, val) => acc + val, 0);
        return sum / data.length;
    }

    // Calculate median
    static median(data) {
        if (data.length === 0) {
            throw new Error("Data array cannot be empty.");
        }
        const sortedData = [...data].sort((a, b) => a - b);
        const mid = Math.floor(sortedData.length / 2);
        return sortedData.length % 2 === 0
            ? (sortedData[mid - 1] + sortedData[mid]) / 2
            : sortedData[mid];
    }

    // Calculate mode
    static mode(data) {
        if (data.length === 0) {
            throw new Error("Data array cannot be empty.");
        }
        const frequency = {};
        data.forEach((value) => {
            frequency[value] = (frequency[value] || 0) + 1;
        });
        const maxFreq = Math.max(...Object.values(frequency));
        const modes = Object.keys(frequency).filter((key) => frequency[key] === maxFreq);
        return modes.length === data.length ? [] : modes.map(Number);
    }

    // Calculate standard deviation
    static standardDeviation(data) {
        if (data.length === 0) {
            throw new Error("Data array cannot be empty.");
        }
        const meanValue = this.mean(data);
        const squaredDiffs = data.map((value) => Math.pow(value - meanValue, 2));
        const variance = this.variance(data);
        return Math.sqrt(variance);
    }

    // Calculate variance
    static variance(data) {
        if (data.length === 0) {
            throw new Error("Data array cannot be empty.");
        }
        const meanValue = this.mean(data);
        const squaredDiffs = data.map((value) => Math.pow(value - meanValue, 2));
        return squaredDiffs.reduce((acc, val) => acc + val, 0) / data.length;
    }

    // Calculate correlation coefficient
    static correlationCoefficient(x, y) {
        if (x.length !== y.length) {
            throw new Error("Data arrays must have the same length.");
        }
        const meanX = this.mean(x);
        const meanY = this.mean(y);
        const covariance = x.reduce((acc, val, idx) => {
            return acc + (val - meanX) * (y[idx] - meanY);
        }, 0);
        const stdDevX = this.standardDeviation(x);
        const stdDevY = this.standardDeviation(y);
        return covariance / (stdDevX * stdDevY * (x.length - 1));
    }

    // Summary statistics
    static summary(data) {
        return {
            mean: this.mean(data),
            median: this.median(data),
            mode: this.mode(data),
            standardDeviation: this.standardDeviation(data),
            variance: this.variance(data),
        };
    }
}

// Example Usage
const dataSet = [1, 2, 2, 3, 4, 5, 5, 5, 6];

// Calculate Mean
const meanValue = Statistics.mean(dataSet);
console.log(`Mean: ${meanValue}`);

// Calculate Median
const medianValue = Statistics.median(dataSet);
console.log(`Median: ${medianValue}`);

// Calculate Mode
const modeValue = Statistics.mode(dataSet);
console.log(`Mode: ${modeValue}`);

// Calculate Standard Deviation
const stdDevValue = Statistics.standardDeviation(dataSet);
console.log(`Standard Deviation: ${stdDevValue}`);

// Calculate Variance
const varianceValue = Statistics.variance(dataSet);
console.log(`Variance: ${varianceValue}`);

// Calculate Correlation Coefficient
const xData = [1, 2, 3, 4, 5];
const yData = [2, 3, 4, 5, 6];
const correlationValue = Statistics.correlationCoefficient(xData, yData);
console.log(`Correlation Coefficient: ${correlationValue}`);

// Get Summary Statistics
const summaryStats = Statistics.summary(dataSet);
console.log('Summary Statistics:', summaryStats);

// math.js - Probability Distributions

class ProbabilityDistributions {
    // Normal Distribution Probability Density Function
    static normalPDF(x, mean = 0, stdDev = 1) {
        const exponent = -((x - mean) ** 2) / (2 * stdDev ** 2);
        return (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
    }

    // Normal Distribution Cumulative Distribution Function
    static normalCDF(x, mean = 0, stdDev = 1) {
        return 0.5 * (1 + this.erf((x - mean) / (stdDev * Math.sqrt(2))));
    }

    // Error Function (used for CDF calculation)
    static erf(x) {
        // Constants
        const a1 =  0.254829592;
        const a2 = -0.284496736;
        const a3 =  1.421413741;
        const a4 = -1.453152027;
        const a5 =  1.061405429;
        const p  =  0.3275911;

        const sign = x >= 0 ? 1 : -1;
        x = Math.abs(x);
        const t = 1.0 / (1.0 + p * x);
        const y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t) * Math.exp(-x * x);
        return sign * (1 - y);
    }

    // Binomial Distribution Probability Mass Function
    static binomialPMF(n, k, p) {
        if (k < 0 || k > n) {
            return 0;
        }
        const coeff = this.binomialCoefficient(n, k);
        return coeff * (p ** k) * ((1 - p) ** (n - k));
    }

    // Binomial Coefficient
    static binomialCoefficient(n, k) {
        if (k > n) return 0;
        if (k === 0 || k === n) return 1;
        let coeff = 1;
        for (let i = 1; i <= k; i++) {
            coeff *= (n - i + 1) / i;
        }
        return coeff;
    }

    // Binomial Distribution Cumulative Distribution Function
    static binomialCDF(n, k, p) {
        let cumulativeProbability = 0;
        for (let i = 0; i <= k; i++) {
            cumulativeProbability += this.binomialPMF(n, i, p);
        }
        return cumulativeProbability;
    }

    // Poisson Distribution Probability Mass Function
    static poissonPMF(k, lambda) {
        return (Math.pow(lambda, k) * Math.exp(-lambda)) / this.factorial(k);
    }

    // Factorial Function
    static factorial(n) {
        if (n < 0) {
            throw new Error("Factorial is not defined for negative numbers.");
        }
        return n <= 1 ? 1 : n * this.factorial(n - 1);
    }

    // Poisson Distribution Cumulative Distribution Function
    static poissonCDF(k, lambda) {
        let cumulativeProbability = 0;
        for (let i = 0; i <= k; i++) {
            cumulativeProbability += this.poissonPMF(i, lambda);
        }
        return cumulativeProbability;
    }

    // Exponential Distribution Probability Density Function
    static exponentialPDF(x, lambda) {
        if (x < 0) {
            return 0;
        }
        return lambda * Math.exp(-lambda * x);
    }

    // Exponential Distribution Cumulative Distribution Function
    static exponentialCDF(x, lambda) {
        if (x < 0) {
            return 0;
        }
        return 1 - Math.exp(-lambda * x);
    }
}

// Example Usage
const mean = 0;
const stdDev = 1;
const xValue = 1;

// Normal Distribution
console.log(`Normal PDF at x=${xValue}: ${ProbabilityDistributions.normalPDF(xValue, mean, stdDev)}`);
console.log(`Normal CDF at x=${xValue}: ${ProbabilityDistributions.normalCDF(xValue, mean, stdDev)}`);

// Binomial Distribution
const n = 10; // number of trials
const k = 5; // number of successes
const p = 0.5; // probability of success
console.log(`Binomial PMF: ${ProbabilityDistributions.binomialPMF(n, k, p)}`);
console.log(`Binomial CDF: ${ProbabilityDistributions.binomialCDF(n, k, p)}`);

// Poisson Distribution
const lambda = 3; // average rate of success
console.log(`Poisson PMF for k=2: ${ProbabilityDistributions.poissonPMF(2, lambda)}`);
console.log(`Poisson CDF for k=2: ${ProbabilityDistributions.poissonCDF(2, lambda)}`);

// Exponential Distribution
const lambdaExp = 1; // rate parameter
const expX = 2; // value for PDF and CDF
console.log(`Exponential PDF at x=${expX}: ${ProbabilityDistributions.exponentialPDF(expX, lambdaExp)}`);
console.log(`Exponential CDF at x=${expX}: ${ProbabilityDistributions.exponentialCDF(expX, lambdaExp)}`);

// math.js - Numerical Integration

class NumericalIntegration {
    // Trapezoidal Rule
    static trapezoidalRule(func, a, b, n) {
        const h = (b - a) / n;
        let sum = 0.5 * (func(a) + func(b));
        
        for (let i = 1; i < n; i++) {
            sum += func(a + i * h);
        }
        
        return sum * h;
    }

    // Simpson's Rule
    static simpsonsRule(func, a, b, n) {
        if (n % 2 !== 0) {
            throw new Error("n must be an even number for Simpson's rule.");
        }

        const h = (b - a) / n;
        let sum = func(a) + func(b);

        for (let i = 1; i < n; i += 2) {
            sum += 4 * func(a + i * h);
        }

        for (let i = 2; i < n - 1; i += 2) {
            sum += 2 * func(a + i * h);
        }

        return (sum * h) / 3;
    }

    // Monte Carlo Integration
    static monteCarloIntegration(func, a, b, n) {
        let sum = 0;

        for (let i = 0; i < n; i++) {
            const x = Math.random() * (b - a) + a;
            sum += func(x);
        }

        return ((b - a) * sum) / n;
    }

    // Example Function to Integrate
    static exampleFunction(x) {
        return Math.sin(x); // Change this function as needed
    }

    // Example Usage
    static exampleUsage() {
        const a = 0; // Lower limit
        const b = Math.PI; // Upper limit
        const n = 1000; // Number of intervals

        console.log("Trapezoidal Rule Result:", this.trapezoidalRule(this.exampleFunction, a, b, n));
        console.log("Simpson's Rule Result:", this.simpsonsRule(this.exampleFunction, a, b, n));
        console.log("Monte Carlo Integration Result:", this.monteCarloIntegration(this.exampleFunction, a, b, n));
    }
}

// Run Example Usage
NumericalIntegration.exampleUsage();
