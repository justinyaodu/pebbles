#ifndef EXPR_H
#define EXPR_H

#include <cstdint>
#include <iostream>
#include <vector>

class Expr {
private:
	static const int32_t AND = -1;
	static const int32_t OR = -2;
	static const int32_t XOR = -3;
	static const int32_t NOT = -4;

	// AND, OR, XOR, NOT, or the number of a variable (e.g. 2 for x2).
	int32_t type;

	// Left and right children, if present.
	Expr* left;
	Expr* right;

	Expr(int32_t type, Expr* left, Expr* right) :
		type(type), left(left), right(right) {}

public:
	// Static helpers to construct various expressions.

	static Expr And(Expr *left, Expr *right) {
		return Expr(Expr::AND, left, right);
	}

	static Expr Or(Expr *left, Expr *right) {
		return Expr(Expr::OR, left, right);
	}

	static Expr Xor(Expr *left, Expr *right) {
		return Expr(Expr::XOR, left, right);
	}

	static Expr Not(Expr *left) {
		return Expr(Expr::NOT, left, nullptr);
	}

	static Expr Var(int32_t var_num) {
		return Expr(var_num, nullptr, nullptr);
	}

	friend std::ostream& operator<< (std::ostream &out, const Expr &expr) {
		switch (expr.type) {
			case Expr::AND:
				return out
					<< "("
					<< *(expr.left)
					<< " && "
					<< *(expr.right)
					<< ")";
			case Expr::OR:
				return out
					<< "("
					<< *(expr.left)
					<< " || "
					<< *(expr.right)
					<< ")";
			case Expr::XOR:
				return out
					<< "("
					<< *(expr.left)
					<< " ^ "
					<< *(expr.right)
					<< ")";
			case Expr::NOT:
				return out
					<< "!"
					<< *(expr.left);
			default:
				return out << "x" << expr.type;
		}
	}

	// Evaluate an expression with the given variable values.
	// The i'th element of vars is the value of the i'th variable.
	bool eval(const std::vector<bool> &vars) {
		switch (this->type) {
			case Expr::AND:
				return this->left->eval(vars) && this->right->eval(vars);
			case Expr::OR:
				return this->left->eval(vars) || this->right->eval(vars);
			case Expr::XOR:
				return this->left->eval(vars) ^ this->right->eval(vars);
			case Expr::NOT:
				return !this->left->eval(vars);
			default:
				return vars[this->type];
		}
	}
};

#endif
