(set-logic BV)
(define-fun Spec ((k1 Bool) (k2 Bool) (k3 Bool) (r1 Bool) (r2 Bool) (r3 Bool) (r4 Bool)) Bool (let ((_let_0 (xor r3 k3))) (let ((_let_1 (xor r2 k2))) (xor (xor (xor (xor (or r2 r3) (xor _let_0 (xor r1 k1))) (and (or (xor r2 r1) (xor _let_0 false)) (or (xor r1 _let_1) (xor _let_0 true)))) r4) (and (and r3 _let_1) (not false))))))
(synth-fun Imp ((k1 Bool) (k2 Bool) (k3 Bool) (r1 Bool) (r2 Bool) (r3 Bool) (r4 Bool)) Bool
((Start Bool) (depth1 Bool) (depth2 Bool) (depth3 Bool) (depth4 Bool) (depth5 Bool) )
((Start Bool ((and depth1 depth1) (or depth1 depth1) (xor depth1 depth1) (not depth1) ))
(depth1 Bool ((and depth2 depth2) (or depth2 depth2) (xor depth2 depth2) (not depth2) ))
(depth2 Bool ((and depth3 depth3) (or depth3 depth3) (xor depth3 depth3) (not depth3) ))
(depth3 Bool ((and depth4 depth4) (or depth4 depth4) (xor depth4 depth4) (not depth4) ))
(depth4 Bool ((and depth5 depth5) (or depth5 depth5) (xor depth5 depth5) (not depth5) ))
(depth5 Bool (k1 k2 k3 r1 r2 r3 r4 ))
))
(declare-var k1 Bool)
(declare-var k2 Bool)
(declare-var k3 Bool)
(declare-var r1 Bool)
(declare-var r2 Bool)
(declare-var r3 Bool)
(declare-var r4 Bool)
(constraint (= (Spec k1 k2 k3 r1 r2 r3 r4) (Imp k1 k2 k3 r1 r2 r3 r4)))
(check-synth)