(define (problem pacman-eat-all)
	
	(:domain pacman)

	(:objects
		sq00 sq01 sq02 - position
	)

	(:init
		(pacman-at-position sq00)
		(food-at-position sq01)
		(food-at-position sq02)
		(adjacent sq00 sq01)  
		(adjacent sq01 sq02)  	
	) 

	(:goal
		(and (not(food-at-position sq01)) 
		     (not(food-at-position sq02))
		     )
	)

)