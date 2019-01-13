# Note on Haskell/FP
Some personal note in learning Haskell or thinking FP

## Mostly thinking or derivation, minimum FP knowledge

### 1. `(.)`
Inspired by the "applicative" chapter in the purple book.  Consider the following **pseudo code**   
`g :: b -> c`  then  
`g . :: (a3 -> b) -> a3 -> c` s.t. `g . f` where `f :: a3 -> b`  then  
`(g .) . :: (a2 -> a3 -> b) -> a2 -> a3 -> c` s.t. `g . f` where `f :: a2 -> a3 -> b` then  
`((g .) .) . :: (a1 -> a2 -> a3 -> b) -> a1 -> a2 -> a3 -> c` s.t. `g . f` where `f :: a1 -> a2 -> a3 -> b`  

Recap **pseudo code**,  
```
  g         ::                    b  ->                   c
  g .       ::             (a3 -> b) ->             a3 -> c
 (g .) .    ::       (a2 -> a3 -> b) ->       a2 -> a3 -> c
((g .) .) . :: (a1 -> a2 -> a3 -> b) -> a1 -> a2 -> a3 -> c
```

#### Factor out `g` to point free styles
Base on the **types** of the previous summary, there is another derivation

`g .` is  
`(.) g` is  
`(.) :: (b -> c) -> (a3 -> b) -> a3 -> c`  

`(g .) .` is  
`((.) g) . ` is  
`(.) ((.) g)` is  
`(.).(.) $ g` is  
`(.).(.) :: (b -> c) -> (a2 -> a3 -> b) -> a2 -> a3 -> c`  

`((g .) .) . ` is  
`((.).(.) $ g) .` is  
`(.) ((.).(.) $ g)` is  
`(.).(.).(.) $ g` is  
`(.).(.).(.) :: (b -> c) -> (a1 -> a2 -> a3 -> b) -> a1 -> a2 -> a3 -> c`

Recap  
```
(.)         :: (b -> c) ->             (a3 -> b) ->             a3 -> c
(.).(.)     :: (b -> c) ->       (a2 -> a3 -> b) ->       a2 -> a3 -> c
(.).(.).(.) :: (b -> c) -> (a1 -> a2 -> a3 -> b) -> a1 -> a2 -> a3 -> c
```

#### Altervative (simpler) way to find the types
Because
```
(.) :: (b -> c) -> (a -> b) -> a -> c
```
Therefore in `(.).(.)`, to compose with the 2nd `(.)`, the 1st `(.)` must be `((a -> b) -> a -> c) -> ???`.  

Comparing the first arguments of both `(.)`s,   
`((a -> b) -> a -> c)` is analogous to `(b -> c)`, substituting `a` by `a1`, `b` by `(a -> b)`, `c` by `a -> c`,  
`???` is `(a1 -> a -> b) -> a1 -> a -> c`, tidy up the variables, it is `(a1 -> a2 -> b) -> a1 -> a2 -> c`

```
(.).(.) :: (b -> c) -> (a1 -> a2 -> b) -> a1 -> a2 -> c
```

### 2. Setter
This is inspired by the lens derivation note.  Suppose  
```
mapOnTraversable :: Traversable t => (a -> b) -> t a -> t b
mapOnTraversable f = runIdentity . traverse (Identity . f)
```

If the data structure is not `Traversable`, we replace `traverse` by another function, say, `setter` such that  
`mapBySetter setter f = runIdentity . setter (Identity . f)` is  
`mapBySetter setter f = (runIdentity . ) setter (Identity . f)` is  
`mapBySetter setter f = (runIdentity . ) . setter . (Identity . ) $ f` is
`mapBySetter setter = (runIdentity . ) . setter . (Identity . )`  
Because  
`(Identity . ) :: (a1 -> a2) -> a1 -> Identity a2`  
`(runIdentity . ) :: (a -> Identity c) -> a -> c`  
Therefore  
`setter :: (a1 -> Identity a2) -> a -> Identity c`  
`mapBySetter :: ((a1 -> Identity a2) -> a -> Identity c) -> (a1 -> a2) -> a -> c`  
rewritten as  
`mapBySetter :: Setter s t a b -> (a -> b) -> s -> t`  
where  
`type Setter s t a b = (a -> Identity b) -> s -> Identity t`  


### 3. `Traversable`
2 main functions:
* `traverse :: Applicative f => (a -> f b) -> t a -> f (t b)`
* `sequenceA :: Applicative f => t (f a) -> f (t a)`

This problem comes from the purple book.
```
data Query = Query
data SomeObj = SomeObj
data IoOnlyObj = IoOnlyObj
data Err = Err

decodeFn = undefined :: String -> Either Err SomeObj

fetchFn = undefined :: Query -> IO [String]

makeIoOnlyObj = undefined :: [SomeObj] -> IO [(SomeObj, IoOnlyObj)]

pipelineFn :: Query -> IO (Either Err [(SomeObj, IoOnlyObj)])
pipelineFn = undefined
```

Purpose is to implement `pipelineFn`

Solve by types, approach:
* starting from `fetchFn`, and `fetchFn`, `decodeFn`, `makeIoOnlyObj` can be "connected" with each other.
* Suppose `:t query` is `Query`
* `:t fetchFn query` is `IO [String]`
* `[]` is `Traversable`, therefore `:t traverse decodeFn <$> fetchFn query` is `IO (Either Err [SomeObj])`
* `IO` is `Monad`, `makeIoOnlyObj` returns `IO`, `Either a` is `Traversable`, therefore `:t traverse decodeFn <$> fetchFn query >>= (traverse makeIoOnlyObj)` is `IO (Either Err [(SomeObj, IoOnlyObj)])`.  Done!
* `fetchFn query` is `IO` and `traverse makeIoOnlyObj` returns `IO`, the implemenation can be re-written as `fetchFn query >>= (traverse makeIoOnlyObj . traverse decodeFn)`
* To take out the `query` for a point-free style, it can be re-written as `(>>= traverse makeIoOnlyObj . traverse decodeFn) . fetchFn`


## Using a type signature to tell if a function is pure or not
A function is pure such that its result only depends on the provided input.  To tell if a function is pure or not, use ```:type```.  If the result type prefixes with ```IO```, this function is impure.  E.g.
```
ghci> :type readFile
readFile :: FilePath -> IO String
```
This IO monad allows you to build any "impure" task such as database update.

## Polymorphic functions
Functions that have type variables are called polymorphic functions.  E.g.
```
ghci> :t head
head :: [a] -> a
```
```a``` is a type variable.  That means ```head``` is a polymorphic function that works on a list of any element types.

## Overloaded functions
Polymorphic functions having the type variables constrainted by type class.  E.g.
```
ghci> :t sum
sum :: Num a => [a] -> a
```

## Pay attention when the name of type constructor and value constructor are the same
People new to FP usually have this mistake.  In defining an ADT, a type constructor and a value constructor have the same name.

Example
```Haskell
data Vector a = Vector a a a deriving (Show)
```
`Vector a` is the **type**, `Vector aaa` is the **value**.  Therefore, `Vector a` but **not** ```Vector a a a``` should be put in the function's type signature.
```Haskell
vplus :: (Num t) => Vector t -> Vector t -> Vector t
(Vector i j k) `vplus` (Vector l m n) = Vector (i+l) (j+m) (k+n)
```

## `newtype` is sometimes used instead of `data`
For a given type, if there is only 1 value constructor which only contains 1 field.  Key word ```newtype``` can be used instead of ```data```.
Example.
```Haskell
data Hm = Hm Int
```
can be written as
```Haskell
newtype Hm = Hm Int
```
The advantage of using ```newtype``` is during runtime, it does not need the wrapping/ unwrapping of the underlying type, ```Int``` in this case.  But during the compilation time, it safesguards the accidental assignment of an ```Int``` value to the ```Hm``` type.
Type variable is allowed in ```newtype```.  Example
```Haskell
newtype Hm a = Hm a
```
```Haskell
newtype Hm a = Hm {getValue :: a}
```

## Reuse type class
Make a type class be subclass of another type class.
```Haskell
class Eq a => Num a where
  -- required behaviour for a Num
```
If ```a``` is ```Num```, it should also be ```Eq```.  i.e ```Num``` is ```Eq```

Type class is used for function declaration, not for data type declaration.  To make a data type be the type class, the type class instance should be defined accordingly.  In some conditions, a type class instance can be defined by re-using another type class instance.  E.g.
```Haskell
instance (Eq m) => Eq (Maybe m) where
  -- implement Eq's required behaviour for the type Maybe m
```
To make ```Maybe m``` be a type class ```Eq```, it should require ```m``` be a type class ```Eq```.  The purpose is similar to requiring a type be a particular type class in defining a function.  E.g.
```Haskell
(+) :: Num a => a -> a -> a
```

## Pragmas
A pragma is a directive to the compiler.  It tells the compiler to enable a language extension that processes input in a way beyond what the standard provides for.

### 1. `forall` and RankNTypes
There is a good explanation in http://sleepomeno.github.io/blog/2014/02/12/Explaining-Haskell-RankNTypes-for-all/

E.g.
```Haskell
applyToTuple :: ([a] -> Int) -> ([b], [c]) -> (Int, Int)
applyToTuple = \f -> \(x, y) -> (f x, f y)
```

It fails to compile because types ```a```, ```b``` and ```c``` do not match.  Theoretically the function of type ```[a] -> Int``` should be applied to any list.  Under this scenario, it needs something call ```RankNTypes```.  Solution:
* Enable GHCI extension
* Use keyword ```forall```

```Haskell
{-# LANGUAGE RankNTypes #-}
applyToTuple :: (forall a. [a] -> Int) -> ([b],[c]) -> (Int, Int)
applyToTuple = \f -> \(x, y) -> (f x, f y)
```

#### Reason
Type signature’s type variables are **implicitly** universally quantified by an **invisible** `forall` section.  Therefore ```applyToTuple :: ([a] -> Int) -> ([b], [c]) -> (Int, Int)``` is actually compiled to ```applyToTuple :: forall a b c. ([a] -> Int) -> ([b], [c]) -> (Int, Int)```

Therefore the type checker expects type variables a, b and c to be different concrete types.  So ```[a] -> Int``` might become ```[Char] -> Int``` or ```[Int] -> Int``` or whatsoever after a function is passed to ```applyToTuple```.  ```(f x, f y)``` seeks to apply that function to two lists of different types – however, any version of that function, i.e. ```[Char] -> Int``` or ```[Int] -> Int``` or whatsoever, expects its list to always be of 1 concrete type only.

If it's re-written as ```applyToTuple :: (forall a. [a] -> Int) -> ([b],[c]) -> (Int, Int)``` which is compiled to ```applyToTuple :: forall b c. (forall a. [a] -> Int) -> ([b],[c]) -> (Int, Int)```, (```a``` and ```b```) or (```a``` and ```c```) will be in **different scopes**.  ```a``` is therefore only 1 type thoughout that function ```[a] -> Int```.  Then this function can be passed any list type.

### 2. `GeneralizedNewtypeDeriving` - make a type class derivable
Suppose there is a type class from which an instance has been created for a type.  If there is a newtype contains this type and we want to reuse this type class instance for this newtype, we can use the pragma ```GeneralizedNewtypeDeriving```.
Example
```Haskell
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

class TooMany a where
  tooMany :: a -> Bool

instance TooMany Int where
  tooMany n = n > 42

newtype Goats = Goats Int deriving (Eq, Show, TooMany)
```

### 3. `InstanceSigs` - allows type signature in defining the type class instance
In defining the type class instance, it is unnecessary and not allowed to write the function type signature.  E.g.
```Haskell
class Functor' f where
  fmap' :: (a -> b) -> f a -> f b

instance Functor' ((->) a) where
  fmap' :: (b -> c) -> (a -> b) -> (a -> c)  -- type signature not allowed
  fmap' = (.)
```
It will raise the error
```
    Illegal type signature in instance declaration:
      fmap' :: (b -> c) -> (a -> b) -> (a -> c)
    (Use InstanceSigs to allow this)
    In the instance declaration for ‘Functor' ((->) a)’
```
To get through the restriction, use `InstanceSigs`.
```Haskell
{-# LANGUAGE InstanceSigs #-}

class Functor' f where
  fmap' :: (a -> b) -> f a -> f b

instance Functor' ((->) a) where
  fmap' :: (b -> c) -> (a -> b) -> (a -> c)
  fmap' = (.)
```

### 4. Data declaration with constraint
Type constraints usually happen on functions.  Sometimes it also happens on a data constructor.  Suppose you want to construct a type by using a type argument where this type belongs to a type class.  E.g.
```Haskell
data WeakenFactor a => Weaken a = Weaken { weakenFactor :: a, weakenHealth :: Health }
```
That means `a` must belong to type class `WeakenFactor`.

Alternatively, it can be presented as a function as follows:
```Haskell
{-# LANGUAGE GADTs #-}  -- this pragama is needed
data Weaken a where -- starts with data, like class
  Weaken :: WeakenFactor a => a -> Health -> Weaken a
```

https://wiki.haskell.org/Data_declaration_with_constraint

### 5. Generalized Algebraic Data Types (GADTs)
GADT can be used to solve the problem mentioned before.  

If we want to define a data constructor in the following way, notice the double colons.
```
data Expr a where
  I   :: Expr Int
```
GADTs extension should be enabled.  

If we want to use smart constructor, it can also apply GADT
```
data Expr a where
  I   :: Int  -> Expr Int
  B   :: Bool -> Expr Bool
  Add :: Expr Int -> Expr Int -> Expr Int
  Mul :: Expr Int -> Expr Int -> Expr Int
  Eq  :: Eq a => Expr a -> Expr a -> Expr Bool
```

In this case, GADT makes use of phantom type but it's more type safe than phantom type because e.g. 
```
I 1 `Eq` B True
``` 
will raise compilation error because `a` can track the type.  For more information, refer to https://en.wikibooks.org/wiki/Haskell/GADT.  

### 6. `RecordWildCards` for `{..}` on data constructor having record syntax
This is for code simplicity.  E.g.
```Haskell
{-# LANGUAGE RecordWildCards #-}
data Stage = Egg { stateChgdTime :: UTCTime,
                   currTemp :: Int ,  -- notice currTemp
                   energyToHatch :: Int,
                   health :: Health }
...
process Egg{..} _ (_, Just IncreaseTemp) allConsts
  | currTemp == (fatalMaxTemp . eggConsts) allConsts = -- currTemp is called directly
    return "The egg has reached the max temperature, you've cooked it"
```
By using `RecordWildCards`, it doesn't need to assign a variable to `Egg` in order to get its current temperature via `currTemp egg`.

### 7. `ViewPatterns`
Enable simple syntax to match a pattern that requires another function to process.  E.g.
```
{-# LANGUAGE ViewPatterns #-}
import qualified Data.Text as T

f :: T.Text -> Either String T.Text
f (T.splitOn ":" -> [_, _]) = Right undefined
f s = Left "input string must be s1:s2 format"
```
The benefit is applied on
```
(T.splitOn ":" -> [_, _])
```
Because
```
T.splitOn ":" :: T.Text -> [T.Text]
```

Note:
* The value to be matched doesn't need to be specified
* Parentheses must be neede


