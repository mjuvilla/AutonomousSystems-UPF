(define (problem pacman-eat-all)
	
	(:domain pacman)

	(:objects
		sq00 sq01 sq02 - position
	)

	(:init
		(pacman-at-position sq01)
		(empty-at-position sq01)
		(adjacent sq00 sq01) 
		(adjacent sq01 sq00)  
		(adjacent sq01 sq02)
		(adjacent sq02 sq01)  	
	) 

	(:goal
		(and (empty-at-position sq00)
			 (empty-at-position sq02)
		   	)
	)

)