#ifndef EXPR_H
#define EXPR_H

#include <algorithm>
#include <cassert>
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
    const int32_t type;

    // Left and right children, if present.
    const Expr* left;
    const Expr* right;

    Expr(const int32_t type, const Expr* left, const Expr* right) :
        type(type), left(left), right(right) {}

public:
    // Static helpers to construct various expressions.

    static const Expr* And(const Expr *left, const Expr *right) {
        return new Expr(Expr::AND, left, right);
    }

    static const Expr* Or(const Expr *left, const Expr *right) {
        return new Expr(Expr::OR, left, right);
    }

    static const Expr* Xor(const Expr *left, const Expr *right) {
        return new Expr(Expr::XOR, left, right);
    }

    static const Expr* Not(const Expr *left) {
        return new Expr(Expr::NOT, left, nullptr);
    }

    static const Expr* Var(int32_t var_num) {
        return new Expr(var_num, nullptr, nullptr);
    }

    void print(std::ostream &out, const std::vector<std::string> *var_names) const {
        switch (type) {
            case Expr::AND:
                out << "(";
                left->print(out, var_names);
                out << " & ";
                right->print(out, var_names);
                out << ")";
                break;
            case Expr::OR:
                out << "(";
                left->print(out, var_names);
                out << " | ";
                right->print(out, var_names);
                out << ")";
                break;
            case Expr::XOR:
                out << "(";
                left->print(out, var_names);
                out << " ^ ";
                right->print(out, var_names);
                out << ")";
                break;
            case Expr::NOT:
                out << "!";
                left->print(out, var_names);
                break;
            default:
                if (var_names == nullptr) {
                    out << "x" << type;
                } else {
                    out << (*var_names)[type];
                }
                break;
        }
    }

    int32_t height(const std::vector<int32_t> &var_heights) const {
        switch (type) {
            case Expr::NOT:
                return 1 + left->height(var_heights);
            case Expr::AND:
            case Expr::OR:
            case Expr::XOR:
                return 1 + std::max(left->height(var_heights), right->height(var_heights));
            default:
                assert(0 <= type && (size_t) type < var_heights.size());
                return var_heights[type];
        }
    }

    const Expr* pad_height(int32_t amount) const {
        assert(amount >= 0);
        switch (amount) {
            case 0:
                return this;
            case 1:
                return Expr::And(this, this);
            default:
                return Expr::Not(Expr::Not(pad_height(amount - 2)));
        }
    }

    const Expr* with_constant_height(int32_t height, const std::vector<int32_t> &var_heights) const {
        const Expr* new_expr;

        switch (type) {
            case Expr::NOT: {
                // Make the operand constant height.
                int32_t left_height = left->height(var_heights);
                const Expr* new_left = left->with_constant_height(left_height, var_heights);

                new_expr = Expr::Not(new_left);
                break;
            }
            case Expr::AND:
            case Expr::OR:
            case Expr::XOR: {
                // Make both operands constant height.
                int32_t left_height = left->height(var_heights);
                const Expr* new_left = left->with_constant_height(left_height, var_heights);
                int32_t right_height = right->height(var_heights);
                const Expr* new_right = right->with_constant_height(right_height, var_heights);

                // Make both operands the same height.
                int32_t child_height = std::max(left_height, right_height);
                new_left = new_left->pad_height(child_height - left_height);
                new_right = new_right->pad_height(child_height - right_height);

                switch (type) {
                    case Expr::AND:
                        new_expr = Expr::And(new_left, new_right);
                        break;
                    case Expr::OR:
                        new_expr = Expr::Or(new_left, new_right);
                        break;
                    case Expr::XOR:
                        new_expr = Expr::Xor(new_left, new_right);
                        break;
                    default:
                        assert(false);
                        break;
                }
                break;
            }
            default:
                // Variables are left as-is.
                assert(0 <= type && (size_t) type < var_heights.size());
                assert(var_heights[type] <= height);
                new_expr = this;
        }

        return new_expr->pad_height(height - new_expr->height(var_heights));
    }

    void assert_constant_height(int32_t height, const std::vector<int32_t> &var_heights) const {
        switch (type) {
            case Expr::NOT:
                left->assert_constant_height(height - 1, var_heights);
                break;
            case Expr::AND:
            case Expr::OR:
            case Expr::XOR:
                left->assert_constant_height(height - 1, var_heights);
                right->assert_constant_height(height - 1, var_heights);
                break;
            default:
                assert(type >= 0 && (size_t) type < var_heights.size());
                assert(var_heights[type] == height);
        }
    }

    friend std::ostream& operator<< (std::ostream &out, const Expr &expr) {
        expr.print(out, nullptr);
        return out;
    }

    // Evaluate an expression with the given variable values.
    // The i'th element of vars is the value of the i'th variable.
    bool eval(const std::vector<bool> &vars) const {
        switch (type) {
            case Expr::AND:
                return left->eval(vars) && right->eval(vars);
            case Expr::OR:
                return left->eval(vars) || right->eval(vars);
            case Expr::XOR:
                return left->eval(vars) ^ right->eval(vars);
            case Expr::NOT:
                return !left->eval(vars);
            default:
                assert(type >= 0 && (size_t) type < vars.size());
                return vars[type];
        }
    }
};

#endif
