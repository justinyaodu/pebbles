;synthesize a countermeasure for the following program
;bool compute( bool k1, bool k2, bool k3, bool r1, bool r2, bool r3, bool r4){
;
;
;  bool n01;
;  bool n02;
;  bool n03;
;  bool n04;
;  bool n05;
;  bool n06;
;  bool n07;
;  bool n08;
;  bool n09;
;  bool n10;
;  bool n11;
;  bool n12;
;  bool n13;
;  bool n14;
;  bool n15;
;  bool n16;
;  bool n17;
;  bool t1;
;  bool t2;
;  bool t3;
;  bool t4;
;  bool t5;
;  bool t6;
;
;
;  n01 = ~r2;
;  t1 = n01 & r3;
;
;
;  n04 = k3 ^ r3;
;  t2 = r2 & n04;
;
;  t3 = r1 ^ 0; 
;
;  t4 = k1 ^ r1;
;
;  n07 = k2 ^ r2;
;  n08 = ~n07;
;  n09 = k3 ^ r3;
;  t5 = n08 & n09;
;
;  n12 = k2 ^ r2;
;  t6 = n12 & r3;
;
;  n14 = t2 ^ t6;
;  n03 = n14 ^ r4;
;
;  n15 = t1 ^ t5;
;  n11 = n03 ^ n15;
;  n16 = n11 ^ t4;
;
;  n17 = n16 ^ t3;
;
;return(n17);
;}

(set-logic BV)

(define-fun Spec ((k1 Bool) (k2 Bool) (k3 Bool) (r1 Bool) (r2 Bool) (r3 Bool) (r4 Bool)) Bool
  (xor (xor r1 false) (xor (xor k1 r1) (xor (xor (xor r3 (not r2)) (and (not (xor k2 r2)) (xor k3 r3))) (xor r4 (xor (and r2 (xor k3 r3)) (and r3 (xor k2 r2)))))))
)
 
(synth-fun Imp ((k1 Bool) (k2 Bool) (k3 Bool) (r1 Bool) (r2 Bool) (r3 Bool) (r4 Bool)) Bool
 ((Start Bool ( 
  (and depth1 depth1) 
  (or depth1 depth1) 
  (xor depth1 depth1) 
  (not depth1) 
  ) )
 (depth1 Bool ( 
  (and depth2 depth2) 
  (or depth2 depth2) 
  (xor depth2 depth2)
  (not depth2) 
  ) )
 (depth2 Bool ( 
  (and depth3 depth3) 
  (or depth3 depth3) 
  (xor depth3 depth3) 
  (not depth3) 
  ) )
 (depth3 Bool ( 
  (and depth4 depth4) 
  (or depth4 depth4) 
  (xor depth4 depth4) 
  (not depth4) 
  ) )
 (depth4 Bool ( 
  (and d5 d5) 
  (or d5 d5) 
  (xor d5 d5)
  (not d5) 
  ) )
 (depth5 Bool ( 
  k1 
  k2 
  k3 
  r1 
  r2 
  r3 
  r4
   ) ) )
)
 
(declare-var k1 Bool)
(declare-var k2 Bool)
(declare-var k3 Bool)
(declare-var r1 Bool)
(declare-var r2 Bool)
(declare-var r3 Bool)
(declare-var r4 Bool)

(constraint (= (Spec k1 k2 k3 r1 r2 r3 r4) (Imp k1 k2 k3 r1 r2 r3 r4)))
(check-synth)

