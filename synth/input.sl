
(set-logic BV)


(define-fun origCir ( (b1 Bool) (b2 Bool) )  Bool    
          (not (and (not b1) ) b2 )
)


(synth-fun skel ( (b1 Bool) (b2 Bool)  )  Bool    
          ((Start Bool (
		                                  (and depth1 depth1)
		                                  (not depth1)
		                                  (or depth1 depth1)
		                                  (xor depth1 depth1)
          ))
          (depth1 Bool (
		                                  (and depth2 depth2)
		                                  (not depth2)
		                                  (or depth2 depth2)
		                                  (xor depth2 depth2)
          ))
          (depth2 Bool (
		                                  (and depth3 depth3)
		                                  (not depth3)
		                                  (or depth3 depth3)
		                                  (xor depth3 depth3)
          ))
          (depth3 Bool (
		                                  b1
										  b2
          ))
)


(declare-var b1 Bool)
(declare-var b2 Bool)

(constraint (= (origCir b1 b2 ) (skel b1 b2 )))


(check-synth)