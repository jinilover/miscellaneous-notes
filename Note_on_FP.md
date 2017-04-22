# Note on FP
It summarizes the key ideas that I think is important when working on FP.  
* I was not major in mathematics.  My perception on the mathematical idea behind FP may not be strictly correct.
* I am new to Haskell.  However, since Haskell is a pure FP language, the FP concept can be understood correctly and more easily using this language.  Therefore I will use Haskell to illustrate the FP concept and then the approach adopted by Scala.

## What is Functional programming (FP)
It is programming in terms of mathematical functions.  Therefore unlike imperative programming, FP incurs no side effect because mathematical functions are simply expressions that produce no side effect.

## Type is important
A mathematical function is proven to be valid only if the argument type is known.  Similarly, the FP code is guranteed to be safe if the expression type is known in compile time http://learnyouahaskell.com/types-and-typeclasses.  Therefore a safe type system is crucial to FP.

## Fundamental concepts in FP

### What is a value, a type and a kind?
e.g. 1 and 2 are **values**.  Both of them belong to the same **type** called ```Integer```.
"a" and "b" are **values**.  Both of them belong to another **type** called ```String```.  
```Integer``` and ```String``` are different **types**.  However, both of them are **types**, therefore both ```Integer``` and ```String``` are the same **kind**.  Later on it will know that a **type** is not the only **kind**.  A **type constructor** is a **kind**.  A **higher kinded type** is another kind **kind**.

### What is type constructor?
As its name implies, it is a constructor that constructs a type.  That is, it constructs a type by reciving a type(s) as arguments.  A **type constructor** is another **kind**.

#### *Type constructor in Haskell*
A type constructor can have 0..n type arguments.  A type constructor hasn't a type argument is just a type.  E.g.
```Haskell
data Bool = True | False
```
Example of a type constructor having a type argument.
```Haskell
data Maybe a = Nothing | Just a
```
Example of a type constructor having 2 type arguments.
```Haskell
data Either a b  =  Left a | Right b
```
In Haskell convention, type variables are written in lower cases.

#### *Type constructor in Scala*
E.g. ```List``` is a type constructor.  It constructs a type ```List[Int]``` when receiving a type argument ```Int```.  A type constructor should be **declared** using an underscore, ```T[_]```, o.w. it is impossible to know ```T``` is a type constructor. But a type constructor should be **referred to** w/o underscore.  This is analogous to declaring and referring to a variable where a type is needed in declaring a variable but the type should be omitted when referring a variable.  Example of declaring and referring to type constructor,
```Scala
object SampleObject {
  def apply[T[_], A: T] = implicitly[T[A]]
}
```
```T``` is a type constructor because it's declared as ```T[_]```.  In ```A: T```, ```T``` is referred to without using an underscore.
To call this function, it should pass a type constructor/a type as the first/second arguments respectively.  E.g.:
```Scala
SampleObject.apply[List, Int]
```
```List``` is a type constructor, ```Int``` is a type.  They satisfy the function argument types requirement.  Please note that ```List``` is passed (aka referred to) without using an underscore.

#### *Value or data constructor in Haskell*
A value constructor can have 0..n arguments.  E.g.
```Haskell
data Maybe a = Nothing | Just a
```
```Nothing``` and ```Just a``` are value constructors.  In this example, 
* ```Maybe``` is a **type constructor**
* ```Nothing``` and ```Just``` are **data or value constructors**.

### Kind
As illustrated before, ```Int``` and  ```String``` are different **types**.  But both of them are **types**.  Therefore they are the same **kind**.  ```Option``` and ```List``` are different **type constructors**.  But they are **type constructors**.  Therefore they are the same **kind**.  And ```Int``` and ```Option``` are different **kinds** because a **type** and a **type constructor** are different **kinds**.

#### *Kind in Haskell*
In ghci, the kind of a type can be checked by the command ```:k```

```Haskell
:k Int
Int :: *
:k Char
Char :: *
```
Therefore ```Int``` and  ```Char``` belong to the same kind ```*```.

```Haskell
:k Maybe
Maybe :: * -> *
:k ([])
([]) :: * -> *
```
```Maybe``` and  ```[]``` belong to the same kind ```* -> *```.

```Haskell
:k Functor
Functor :: (* -> *) -> Constraint
:k Monad
Monad :: (* -> *) -> Constraint
```
This time the kind is ```(* -> *) -> Constraint```

#### *Kind in Scala*
Similar ideas are conveyed in Scala.  E.g. 
* ```List[_]``` and ```Option[_]``` are of the same kind ```* -> *```.  
* ```Map[_, _]``` and ```Function1[_, _]``` are of the same kind ```* -> * -> *```.
* ```Functor[F[_]]``` and ```Monad[M[_]]``` are of the same kind ```(* -> *) -> *```.

### Higher kinded type
In the previous examples, there is a kind look like ```(* -> *) -> ...```.  Notice the parentheses.  It looks like a higher-order function such as ```filter :: (a -> Bool) -> [a] -> [a]```.  This is called a higher kinded type.  Higher kinded types are type constructors that take type constructor as argument to constructor a new type. http://raichoo.blogspot.com.au/2011/07/from-functions-to-monads-in-scala.html.  
For example, ```FoldMappable``` is a higher kinded type
```Scala
trait FoldMappable[T[_]] {
  def foldMap[A, B: Monoid](fm: T[A])(f: A => B): B
}
```

Summary: 
* There are different **kinds** such as **type**, **type constructor**, **higher kinded type** and so on.
* ```Int```, ```Long```, ```Maybe Char``` belong to the same **kind** because they are **types** represented by ```*```
* ```List```, ```[]```, ```Maybe``` belong to the same **kind** because they are **type constructors** represented by ```* -> *```
* ```Functor```, ```Applicative```, ```Monad``` belong to the same **kind** because they are **higher kinded types** represented by ```(* -> *) -> *```

### Type class
A type class is an interface that defines some behaviour.  When a type is a **part** of a type class, this type supports the behaviour defined by this type class.  A type class is not the same as a class or interface in the OO world.
Consider the following Haskell function ```(==)```:
```Haskell
:t (==)
(==) :: (Eq a) => a -> a -> Bool
```
That means, to be able to use the function ```(==)```, the type of the 2 arguments should support the behaviour of the type class ```Eq```.  Alternative speaking, ```Eq``` is the class constraint on the function ```(==)``` arguments.

#### *How to use type class in Haskell*
Suppose I want to define a function ```plus``` that operates on 2 arguments whose types should be part of type class ```Number```, type class ```Number``` should be defined as:
```Haskell
class Number a where
	plus :: a -> a -> a
```
After defining this type class, function ```plus``` is available.  However, to use ```plus```, the argument type must be part of type class ```Number```.  E.g. if I want ```plus``` to operate on 2 integers, the type ```Integer``` must be part of ```Number``` by defining ```Integer``` as a ```Number``` instance.
```Haskell
instance Number Integer where
	plus = \x y -> x + y
```
Please note that the ```plus``` implementation detail is not important.  After implementing ```Integer``` as part of ```Number```, I can use ```plus```  to operate on 2 integers as, say, ```plus 1 2``` which behaves according to the ```plus``` implementation in ```instance Number Integer```

#### *How to use type class in Scala*
Scala is a JVM language.  It does not have this inborn type class feature as in a pure FP language.  To implement this feature, ```trait``` is used.
```Scala
trait Number[A] {
  def plus(a1: A, a2: A): A
}
```
To make this "function" ```plus``` operate on 2 integers, type ```Int``` must be "part" of ```Number``` by defining ```Int``` as a ```Number``` instance - ``Number[Int]```.
```Scala
implicit object IntNumber extends Number[Int] {
  def plus(i1: Int, i2: Int): Int = i1 + i2
}
```
After importing this implicit object, "function" ```plus``` can now operate on 2 integers as ```implicitly[Number[Int]].plus(1, 2)```

If I want to parameterize the type, it can be done as
```Scala
def sum[A](a1: A, a2: A)(implicit p: Number[A]): A = p.plus(a1, a2)
```
or
```Scala
def sum[A: Number](a1: A, a2: A): A = implicitly[Number[A]].plus(a1, a2)
```
```[A: Number]``` is known as a **context bound**.  It states that ```Number``` is a type class for ```A``` whose ```Number[A]``` instance is implicitly imported to make ```implicitly[Number[A]]``` work.

#### *Type class vs OO polymorphism*
As mentioned before, a type class is neither a class nor interface as in the OO world. 
* OO's polymorphism is about class inheritance or interface implementation.  Type class is about **parametric polymorphism**.
* OO's polymorphism considers from a **class**/**interface**'s point of view.  Therefore ```plus``` will be declared on the interface as ```def plus(a: A): A``` and thus implemented on the class as ```def plus(i: Int): Int = blah```
* Type class considers from a **function**'s point of view.  ```plus``` is a **standalone** function that operates on arguments of required type class.  The type that implements the required type class behaviour defines how ```plus``` behaves when operating on arguments of this type.

#### *Type class is important in FP*
The example illustrates how to use a type class to define the behaviour of a type.  A type class is not limited to defining the behaviour of types but other kinds as well.  E.g. type class ```Functor``` defines the behaviour for a type constructor such as ```Maybe```.  Therefore type class is very important for defining many useful abstractions.

## Useful abstractions

### Monoid
There is a number of types that can be "added" and an identity element which give out the following laws:
* The element "addition" is associative
* The "addition" of an element to an identity element results in the same element

#### *Monoid in Haskell*
```Haskell
class Monoid a where
	mempty :: a
	mappend :: a -> a -> a
```

To make list ```[a]``` be a monoid.
```Haskell
instance Monoid [a] where
    mempty = []
    mappend = \xs1 -> \xs2 -> xs1 ++ xs2
```

To test it
```Haskell
mempty :: [a]
[]

mappend [1,2,3] [3,4,5]
[1,2,3,3,4,5]
```

#### *Monoid in Scala*
```Scala
trait Monoid[A] {
  def mzero: A
  def append(a1: A, a2: A): A
}
```
To make ```List[Int]``` be a monoid.
```Scala
object Implicits {
  implicit def listMonoid[A] = new Monoid[List[A]] {
    def mzero: List[A] = List.empty[A]
    def append(a1: List[A], a2: List[A]): List[A] = a1 ++ a2
  }
}
```
To test it
```Scala
import Implicits._
implicitly[Monoid[List[Int]]]
```
A companion object can be defined for convenience.
```Scala
object Monoid {
  def apply[A: Monoid]: Monoid[A] = implicitly[Monoid[A]]
}
```
to test it
```Scala
import Implicits._
Monoid[List[Int]]
```
The ```Monoid``` trait and a number of ```Monoid``` instances are available from Scalaz.
```Scala
import scalaz.Monoid
def foldMap[A, B: Monoid](it: List[A])(f: (A) => B): B = {
	val m = implicitly[Monoid[B]]
    it.foldLeft(m.zero) {
		(z, a) => m.append(z, f(a))
	}
}

import scalaz.std.anyVal._  // import implicit Monoid[Int] instance
foldMap(List(1,2,3))(_ * 2)
```
Using ```m.zero``` and ```m.append``` may be cumbersome.  Scalaz provides ```scalaz.syntax.monoid._```, an implicit conversion that simplifies ```foldMap``` as follows.
```Scala
import scalaz.syntax.monoid._
import scalaz.Monoid
def foldMap[A, B: Monoid](it: List[A])(f: (A) => B): B =
	it.foldLeft(mzero[B]) {
    	(z, a) => z |+| f(a)
	}
```

#### *Note about Monoid*
* Monoid is a type class that defines class constraint for a **type**.  This can be understood according to the Monoid definition.  E.g. a Monoid instance can be defined for a type ```List[A]``` but not for a type constructor ```List```.
* In the Scala implementation, trait ```Monoid``` and ```implicit``` are means to help to define an Monoid instance for, say, ```List[Int]```.  The reason is Scala is not a pure FP language.   It doesn't have an inborn type class feature.  Therefore in terms of FP terminology, ```List[Int]``` is the Monoid, ```Monoid[List[Int]]``` is just a means to make ```List[Int]``` a monoid.

### Functor
Suppose 
* There is a function or morphism ```f: A => B```.
* There is a type constructor ```F[_]```.

If there is a morphism that manipulates type ```A``` (where ```A``` can be Int, String, ...) in the first category and there is a transformed morphism that manipulates the transformed type ```F[A]``` (i.e. F[Int], F[String], ...) in the second category, then this transformation ```F[_]``` is a **functor**.

#### *Functor in Haskell*
```Haskell
class Functor f where
    fmap :: (a -> b) -> f a -> f b
```
To make ```[]``` be a functor
```Haskell
instance Functor [] where
    fmap = map
```
To test it
```Haskell
fmap (\x -> x * 2) [1,2,3]
[2,4,6]
```

#### *Functor in Scala*
```Scala
trait Functor[F[_]] {
    def map[A,B](x : F[A])(f: A=>B) : F[B]
}
```
To make ```List``` be a functor
```Scala
object Implicits {
  implicit object ListFunctor extends Functor[List] {
    def map[A, B](x: List[A])(f: A => B): List[B] =
      x map f
  }
}
```
To test it
```Scala
import Implicits._
implicitly[Functor[List]].map(List(1,2,3))(_ * 2)
List[Int] = List(2, 4, 6)
```
The ```Functor``` trait, companion object and a number of ```Functor``` instances are available from Scalaz.
```Scala
import scalaz.std.list._  // import implicit Functor[List] instance
Functor[List].map(List('a', 'b', 'c'))(_.toUpper)
```

#### *The 2 Functor laws*
The laws are illustrated in Haskell only.  The Scala counterparts can be imagined correspondingly.

##### *Law 1 - Identity*
```Hasekll
fmap id = id
```
where ```id``` is the identity function.

##### *Law 2 - Composition*
```Hasekll
fmap f . fmap g = fmap (f . g)
```

#### *Note about Functor*
* A functor is a type class that defines class constraint for a **type constructor**.
* Similar to Monoid, in the Scala implementation, ```List``` is the Functor, not the ```Functor[List]``` instance.
* A functor is also the morphism of functions between categories.  The reason is a functor maps a function of ```A => B``` (```A``` and ```B``` belongs to 1 category) to another function of ```F[A] => F[B]``` (where ```F[A]``` and ```F[B]``` belongs to another category).

### Monad
A Monad is a Functor.  It has 2 more functions
* point or return
* bind or >>=

#### *Monad in Haskell*
```Haskell
class Monad m where
    return :: a -> m a
    (>>=) :: m a -> (a -> m b) -> m b
```
To make ```[]``` a monad.
```Haskell
instance Monad [] where
    return = \a -> [a]
    (>>=) = \xs -> \f -> concat . map f $ xs
```
To test it
```Haskell
(return 'x') :: [Char]
"x"

[1,2,3] >>= (\x -> return $ x * 2)
[2,4,6]
```

#### *do-notation in Haskell*
do-notation is a syntactic sugar to improve readability on using ```>>=``` and lambdas.
Example
```Haskell
foldl (\z -> \x -> z >>= (\zVal -> fmap (\xVal -> xVal + zVal) x)) (Just 0) [Just 2, Just 3, Just (-1)]
```
can be written as
```Haskell
foldl (\z -> \x -> 
        do 
            zVal <- z
            xVal <- x
            return $ zVal + xVal
    ) (Just 0) [Just 2, Just 3, Just (-1)]
```

#### *Monad in Scala*
```Scala
trait Monad[M[_]] {
	def point[A](value: A): M[A]
    def bind[A, B](value: M[A])(f: A => M[B]): M[B]
}
```
To make ```List``` a monad.
```Scala
object Implicits {
  implicit object ListMonad extends Monad[List] {
    def point[A](value: A): List[A] = List(value)
    def bind[A, B](value: List[A])(f: A => List[B]): List[B] =
      value flatMap f
  }
}
```
To test it
```Scala
import Implicits._
implicitly[Monad[List]].bind(List(1,2,3))(x => List(x * 2))

List[Int] = List(2, 4, 6)
```
The ```Monad``` trait, companion object and a number of ```Monad``` instances are available from Scalaz.

The following example is similar to the ```foldMap``` example in the ```Monoid``` section.  This time it performs monadiac operation such that the return type is ```M[B]``` instead of ```[B]```.
```Scala
import scalaz.{Monoid, Monad}
import scalaz.syntax.monoid._
import scalaz.std.anyVal._

def foldMapM[A, B: Monoid, M[_] : Monad](it: List[A])(f: A => M[B]): M[B] = {
  val m = implicitly[Monad[M]]
  it.foldLeft(m point mzero[B]) {
    (z, a) =>
      m.bind(z){
        zVal => m.map(f(a))(zVal |+| _)
      }
  }
}
```
Using ```m.point```, ```m.bind``` and ```m.map``` may be cumbersome. Scalaz provides ```scalaz.syntax.monad._```, an implicit conversion that simplifies the code as follows:
```Scala
import scalaz.{Monoid, Monad}
import scalaz.syntax.monoid._
import scalaz.syntax.monad._

def foldMapM[A, B: Monoid, M[_] : Monad](it: List[A])(f: A => M[B]): M[B] =
  it.foldLeft(mzero[B].point[M]) {
    (z, a) =>
      z flatMap {
        zVal => f(a) map (zVal |+| _)
      }
  }
```
```scalaz.syntax.monad._``` also enable to use for-notation as follows:
```Scala
import scalaz.{Monoid, Monad}
import scalaz.syntax.monoid._
import scalaz.syntax.monad._
def foldMapM[A, B: Monoid, M[_] : Monad](it: List[A])(f: A => M[B]): M[B] = 
    it.foldLeft(mzero[B].point[M]) {
		(z, a) =>
			for {
            	zVal <- z
                bVal <- f(a)
            } yield zVal |+| bVal
    }
```

#### *Note about Monad*
* A monad is a type class that defines class constraint for a **type constructor**.
* Monads are more commonly used than Functors.  The reason is ```bind```  or ```>>=``` enables sequencing computation.  That is, the function inside ```bind``` is ```A => M[B]``` and ```M[B]``` is the result of another monadic operation.
* Scala's for-notation is similar to Haskell's do-notation.  The slight difference is ```yield``` should be used together with ```for``` such that the value inside ```yield``` should be a value be wrapped by the Monad.  In ```do``` notation, there is nothing analogous to ```yield```.  The last expression should be a Monad to be returned from this do-notation expression.

### Applicative
An applicative is a functor.  A monad is an applicative.  It has the function <*> or ap.

#### *Applicative in Haskell*
```Haskell
class Applicative m where 
    (<*>) :: m (a -> b) -> m a -> m b
```
To make ```[]``` an applicative.
```Haskell
instance Applicative [] where
    (<*>) fs xs = [f x | f <- fs, x <- xs]
```
Example 1
```Haskell
[(+1), (+2)] <*> [1,2,3]

[2,3,4,3,4,5]
```
It can be re-written using Monad function ```>>=``` as follows
```Haskell
[(+1), (+2)] >>= (<$> [1,2,3])
```
Or re-written using do-notation as follows
```Haskell
do
  f <- [(+1), (+2)]
  x <- [1,2,3]
  return $ f x
```

Example 2
```Haskell
(,) <$> (Just 3) <*> (Just True)

Just (3,True)
```
Again, it can be re-written using Monad function ```>>=``` as follows
```Haskell
(,) <$> Just 3 >>= (<$> Just True)
```

#### *Applicative in Scala*
```Scala
trait Applicative[F[_]] {
  def ap[A, B](value: F[A])(func: => F[A => B]): F[B]
}
```
To make ```List``` an applicative.
```Scala
object Implicits {
  implicit object ListApplicative extends Applicative[List] {
    def ap[A, B](value: List[A])(func: => List[A => B]): List[B] =
      func flatMap(f => value map f)
  }
}
```
To test it
```Scala
import Implicits._
implicitly[Applicative[List]].ap[Int, Int](List(1,2,3))(List(_ + 1, _ + 2))
```
Consider the following example
```Scala
def optionSequencing[T](optA: Option[T], optB: Option[T], optC: Option[T])(f: (T, T, T) => T): Option[T] =
    for {
        a <- optA
        b <- optB
        c <- optC
    } yield f(a, b, c)
```
```optionSequencing``` can be implemented by alternative approach via Applicative.
```Option``` is a Monad which is therefore an Applicative.  Assume there is an Applicative[Option] instance.
```Scala
def optionSequencing_ap[T](optA: Option[T], optB: Option[T], optC: Option[T])(f: (T, T, T) => T): Option[T] = {
    val appl = implicitly[Applicative[Option]]
    val f1: Option[T => (T => (T => T))] = Some(f.curried)
    val inter1: Option[T => (T => T)] = appl.ap(optA)(f1)
    val inter2: Option[T => T] = appl.ap(optB)(inter1)
    appl.ap(optC)(inter2)
}
```
It can be simplified.
```Scala
def optionSequencing_ap[T](optA: Option[T], optB: Option[T], optC: Option[T])(f: (T, T, T) => T): Option[T] = {
    val appl = implicitly[Applicative[Option]]
	appl.ap(optC)(appl.ap(optB)(appl.ap(optA)(Some(f.curried))))
```
In this scenario, I can see using Applicative is more flexible than using Monad.

#### *The 4 Applicative laws*
The laws are illustrated in Haskell only.  The Scala counterparts can be imagined easily.

##### *Law 1 - Identity*
```Hasekll
pure id <*> a = a
```

##### *Law 2 - Homomorphism*
```Haskell
pure f <*> pure a = pure $ f a
```

##### *Law 3 - Interchange*
```Haskell
f <*> pure a = pure ($ a) <*> f
```

```f <*> pure a``` can be written as
```pure (\f -> f a) <*> f``` can be simplified as
```pure ($ a) <*> f```

##### *Law 4 - Composition*
```Haskell
pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
```

Explanation:
```(.) :: (b -> c) -> (a -> b) -> a -> c```

**Imagine** ```u :: f (b -> c)```
Then ```pure (.) <*> u :: f ((a -> b) -> a -> c)```

**Imagine** ```v :: f (a -> b)```
Then ```pure (.) <*> u <*> v :: f (a -> c)```

**Imagine** ```w :: f a```
Then ```pure (.) <*> u <*> v <*> w :: f c```

Considering the types of u, v and w, ```u <*> (v <*> w) :: f c```, therefore
```Haskell
pure (.) <*> u <*> v <*> w :: u <*> (v <*> w) :: f c
```

#### *Note about Applicative*
* An applicative is a type class that defines class constraint for a **type constructor**.
* As illustrated in the 2 Haskell examples, they can be implemented using monad.  Using applicative makes the code simpler.  It is the reason why applicative is used in the scenarios when a function is enclosed inside a monad.

## Design Patterns

### Free monad
The idea is making a sequencing computation where this sequence is called "AST".  This "AST" is presented in the form of executing Free Monads in sequence via flatMap/map or for-exp.  Each sequenced Free Monad only declares what should be done.  Maybe that's why it's "Free".  How to fill in the business logic for the Free Monads is left to the "interpreter".  This is analogous to the polymorphism in the OO design.
In Scalaz, a Free Monad looks like
```Scala
Free[F[_], A]
```
```F``` is a Functor.  ```A``` is the data type such that there is a transformed data type ```F[A]``` where morphism is reserved.

Here a good example of learning Free Monad.  https://github.com/kenbot/free/blob/master/src/main/scala/kenbot/free/KVS.scala

#### "AST" duties - what should be done
In the example, the sequence of executing the Free Monads, or called "script", is
```Scala
val script: Free[KVS, Unit] = for {
    id <- get("swiss-bank-account-id")
    _ <- modify(id, (_ + 1000000))
    _ <- put("bermuda-airport", "getaway car")
    _ <- delete("tax-records")
} yield ()
```
The Free Monads are:
* ```get("swiss-bank-account-id")``` - underlying is ```Free.liftF(Get("swiss-bank-account-id", Predef.identity))```
* ```put("bermuda-airport", "getaway car")``` - underlying is ```Free.liftF(Put("bermuda-airport", "getaway car", ()))```
* ```modify(id, (_ + 1000000))``` - a Free Monad composed of ```get(id)``` and ```put(id, (_ + 1000000)(someValue))```
* ```delete("tax-records")``` - underlying is ```Free.liftF(Delete("tax-records", ()))```

As the name implies, ```Free.liftF``` lifts a functor to a Free Monad.  Therefore ```Get```, ```Put```, ```Delete``` are functors.  Here is the "inheritance" hierarchy of the functors. 
```Scala
sealed trait KVS[+Next]
case class Put[Next](key: String, value: String, next: Next) extends KVS[Next]
case class Get[Next](key: String, onResult: String => Next) extends KVS[Next]
case class Delete[Next](key: String, next: Next) extends KVS[Next]
```
Programmically these Scala classes are functors.  In terms of high level FP design, they are algebraic data types (ADT).  An ADT is not necessarily a functor.  In an FP design using Free Monad, a functor is a means to implement some of the ADTs.  

In this example, we can tell ```Next``` is the data type be transformed by the functor.  It is also the data type worked by the Free Monad.

Since ```KVS``` is a functor, a ```Functor[KVS]``` should be implemented as follows.
```Scala
new Functor[KVS] {
    def map[A,B](kvs: KVS[A])(f: A => B): KVS[B] = kvs match {
      case Put(key, value, next) => Put(key, value, f(next))
      case Get(key, onResult) => Get(key, onResult andThen f)
      case Delete(key, next) => Delete(key, f(next))
    }
```
The ```Functor[KVS]``` implementation is pretty straight forward, there will be similar implementation for all other functors in Free Monads.  To save the typing, Scalaz provides an alternative approach ```Coyoneda[KVS, A]``` http://docs.typelevel.org/api/scalaz/nightly/#scalaz.Coyoneda.  https://github.com/jinilover/scalaForFun/tree/master/src/main/scala/battleship illustrates how to use ```Coyoneda[KVS, A]```

We have identitified what tasks should be done in the form of Free Monads and the sequence of executing them.  The implementation of these Free Monads and the concrete type of ```Next``` are left to the "interpreter".

#### "Interpreter" duties - how to do it
An interepreter is responsible to fill in the Free Monad business logic and drive the sequenced Free Monads.  Scalaz provides different ways to fill in the business logic such as ```Free.resume.fold``` (which involves recursion), ```Free.runFC(script)(naturalTransformation``` and many other ways.  This example uses ```Free.resume.fold``` that implements the Free Monad as follows:
```Scala
def interpretPure(kvs: Script[Unit], table: Map[String, String] = Map.empty): Map[String, String] = kvs.resume.fold({
    case Get(key, onResult) => interpretPure(onResult(table(key)), table)
    case Put(key, value, next) => interpretPure(next, table + (key -> value))
    case Delete(key, next) => interpretPure(next, table - key)
  }, _ => table)
```
The interpreter also defines the concrete data type for ```Next```.  According to this implementation and how Free Monads are declared in ```Free.liftF```, the concrete types of ```Next``` are:
* ```String``` in ```Get[Next](key: String, onResult: String => Next)```
* ```Unit``` in ```Put[Next](key: String, value: String, next: Next)```
* ```Unit``` in ```Delete[Next](key: String, next: Next)```

#### Recap
There are different ways to implement Free Monad using Scalaz.  The basic steps are:
* Identifying the tasks to be executed, that is, identifying the ADTs in the form of a functor hierarchy.
* Lifting these functors to Free Monads
* Declaring the sequence of the Free Monad execution.  That is, using flatMap/map or for-exp.
* Creating a ```Functor[KVS]``` instance or using ```Coyoneda[KVS, A]```
* Filling in the business logic for the Free Monad using ```Free.resume.fold``` or ```Free.runFC(script)(naturalTransformation```.
* Drive the script execution.

https://github.com/jinilover/scalaForFun/tree/master/src/main/scala/battleship illustrates how to use ```Coyoneda[KVS, A]``` (to save the boilerplate of creating a ```Functor[KVS]``` instance) and Natural transformation to fill in the Free Monad business logic.

#### Free Monad internal mechanism
I don't completely understand how Free Monad works.  The features I am sure is:
* A Free Monad is a monad.  Therefore it should have ```flatMap``` and ```map``` defined.
* In defining an ```Functor[_]``` instance, the func ```f: A => B``` is the key that makes the Free Monad sequencing work via for-exp (or flatMap/map).


