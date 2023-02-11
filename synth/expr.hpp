#ifndef EXPR_H
#define EXPR_H

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
    int32_t type;

    // Left and right children, if present.
    Expr* left;
    Expr* right;

    Expr(int32_t type, Expr* left, Expr* right) :
        type(type), left(left), right(right) {}

public:
    // Static helpers to construct various expressions.

    static Expr* And(Expr *left, Expr *right) {
        return new Expr(Expr::AND, left, right);
    }

    static Expr* Or(Expr *left, Expr *right) {
        return new Expr(Expr::OR, left, right);
    }

    static Expr* Xor(Expr *left, Expr *right) {
        return new Expr(Expr::XOR, left, right);
    }

    static Expr* Not(Expr *left) {
        return new Expr(Expr::NOT, left, nullptr);
    }

    static Expr* Var(int32_t var_num) {
        return new Expr(var_num, nullptr, nullptr);
    }

    void print(std::ostream &out, std::vector<std::string> *var_names) const {
        switch (type) {
            case Expr::AND:
                out << "(";
                left->print(out, var_names);
                out << " && ";
                right->print(out, var_names);
                out << ")";
                break;
            case Expr::OR:
                out << "(";
                left->print(out, var_names);
                out << " || ";
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

    void assert_depth(uint32_t depth, const std::vector<uint32_t> &var_depths) {
        switch (type) {
            case Expr::NOT:
                left->assert_depth(depth - 1, var_depths);
                break;
            case Expr::AND:
            case Expr::OR:
            case Expr::XOR:
                left->assert_depth(depth - 1, var_depths);
                right->assert_depth(depth - 1, var_depths);
                break;
            default:
                assert(type >= 0 && (size_t) type < var_depths.size());
                assert(var_depths[type] == depth);
        }
    }

    friend std::ostream& operator<< (std::ostream &out, const Expr &expr) {
        expr.print(out, nullptr);
        return out;
    }

    // Evaluate an expression with the given variable values.
    // The i'th element of vars is the value of the i'th variable.
    bool eval(const std::vector<bool> &vars) {
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
