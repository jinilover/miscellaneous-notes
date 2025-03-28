# Note on Haskell/FP
Personal note in Haskell/FP, most of them concluded from reasoning instead of Haskell specific knowledge.

### 1. `(.)`
Inspired by the "applicative" chapter in the purple book.  Consider the following **pseudo code**   
`g :: b -> c`  
`g . :: (a -> b) -> a -> c` imagine there is `f :: a -> b` s.t. `g . f`    
`(g .) . :: (a1 -> a2 -> b) -> a1 -> a2 -> c` imagine there is `f :: a1 -> a2 -> b`  
`((g .) .) . :: (a1 -> a2 -> a3 -> b) -> a1 -> a2 -> a3 -> c`  

Recap **pseudo code**,  
```
  g         ::                    b  ->                   c
  g .       ::             (a3 -> b) ->             a3 -> c
 (g .) .    ::       (a2 -> a3 -> b) ->       a2 -> a3 -> c
((g .) .) . :: (a1 -> a2 -> a3 -> b) -> a1 -> a2 -> a3 -> c
```

#### Factor out `g` to point free styles
According to the previous findings, we can deduce the type of `(.).(.)`

`(g .) .` is  
`((.) g) . ` is  
`(.) ((.) g)` is  
`(.).(.) $ g`, factoring out `g` gives   
`(.).(.) :: (b -> c) -> (a1 -> a2 -> b) -> a1 -> a2 -> c`  

Similarly  
`((g .) .) . ` is  
`((.).(.) $ g) .` is  
`(.) ((.).(.) $ g)` is  
`(.).(.).(.) $ g` is  
`(.).(.).(.) :: (b -> c) -> (a1 -> a2 -> a3 -> b) -> a1 -> a2 -> a3 -> c`

Another discovery  
```
(.)         :: (b -> c) ->             (a3 -> b) ->             a3 -> c
(.).(.)     :: (b -> c) ->       (a2 -> a3 -> b) ->       a2 -> a3 -> c
(.).(.).(.) :: (b -> c) -> (a1 -> a2 -> a3 -> b) -> a1 -> a2 -> a3 -> c
```

#### Altervative way to find the types w/o using `g`
Because
```
(.) :: (b -> c) -> (a -> b) -> a -> c
```
Therefore in `(.).(.)`, to compose with the right `(.)`, the left `(.)` must be `((a -> b) -> a -> c) -> ???`.  

Since `(.)` must be something like `(b -> c) -> (a -> b) -> a -> c`,

compare `((a -> b) -> a -> c) -> ???` with `(b -> c) -> (a -> b) -> a -> c`,

`((a -> b) -> a -> c)` is analogous to `(b -> c)`.

Therefore `???` is `(a1 -> a -> b) -> a1 -> a -> c` which can be tidied as `(a1 -> a2 -> b) -> a1 -> a2 -> c`.
So `(.).(.)` is `(b -> c) -> (a1 -> a2 -> b) -> a1 -> a2 -> c`.

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


## `IORef`, `STRef`
Both achive mutability in side-effect free manner using different means.

### `IORef`
`IORef` provides mutable value in `IO` monad.  As the name implies, it's reference to some data that allows functions to change this data and these functions must operate in `IO`.  That is, the data can only be modified by going through `IO`.

### `STRef`
* `STRef` provdes mutable value in `ST s` monad where the mutable memory is only used internally, i.e. the functions must be operate in `ST s`.  
* `s` represents the mutable memory which is set by library or GHC developers.
* Function such as `runST :: (forall s. ST s a) -> a` uses rank-2 polymorphism.  This implies the function implementation determines `s`.  It's unlikely to set `s` at the application level.  
* The monadic context is `ST s` and since it's unlikely to have different `s` on the application level, it's unlikely have `ST s1` and `ST s2` in monadic operation to perform the data mutation.

References:
* https://wiki.haskell.org/Monad/ST
* https://stackoverflow.com/questions/5545517/difference-between-state-st-ioref-and-mvar?rq=1

## `MVar`
`MVar` is used for multi threading, i.e. `MVar a` is used to pass information between threads.  As analogy, it can be regarded as `IORef` with locks.

References:
* https://riptutorial.com/haskell/example/15506/communicating-between-threads-with--mvar-
* https://stackoverflow.com/questions/15439966/when-why-use-an-mvar-over-a-tvar

## `ByteString`
It's a space-efficient representation of a `Word8` vector.  It is not intended to be a string and is an array of bytes that comes in both strict and lazy forms.  It can be used to store 8-bit character strings and also to handle binary data.   This type is not for text manipulation as it doesn’t support Unicode. It's de facto standard type for networking, serialization, and parsing in Haskell.

## `Text`
`Text` provides an efficient packed Unicode text type.

## `String` vs `Text`
If the application targets at processing large amounts of character oriented data and/or various encodings, then it should use `Text`.  Otherwise, using `String` is fine.

References:
* http://www.alexeyshmalko.com/2015/haskell-string-types/
* https://mmhaskell.com/blog/2017/5/15/untangling-haskells-strings

## Performance/Strictness/Laziness
Because of laziness, the compiler can't evaluate a function argument and pass the value to the function, it has to record the expression in the heap in a suspension (or thunk) in case it is evaluated later. Storing and evaluating suspensions is costly, and unnecessary if the expression was going to be evaluated anyway.

References:
* https://wiki.haskell.org/Performance/Strictness

## Using a type signature to tell if a function is pure or not
A function is pure such that its result only depends on the provided input.  To tell if a function is pure or not, use ```:type```.  If the result type prefixes with ```IO```, this function is impure.  E.g.
```
ghci> :type readFile
readFile :: FilePath -> IO String
```
This IO monad allows you to build any "impure" task such as database update.

## Variance, positive/negative position

Positive position means the type variable is the **result/output/range/codomain** of the function, e.g. `Maybe a`, `r -> a`.  Under this condition, the data type is covariant with `a` s.t. a **functor** instance, but not contravariant instance, can be made for the "context" of `a`

Negative position means the type variable is the **argument/input/domain** of the function, e.g. `data MakeInt a = MakeInt (a -> Int)`.  Under this condition, the data type is contravariant with `a` s.t. a **contravariant** instance, but not functor instance, can be made for the "context" of `a`

Reference:
* https://www.fpcomplete.com/blog/2016/11/covariance-contravariance 

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
People new to FP usually have this mistake.  Imagine a type constructor and a value constructor have the same name as follows:

Example
```Haskell
data Vector a = Vector a a a deriving (Show)
```
`Vector a` is the **type**, `Vector a a a` is the **value**.  Therefore, `Vector a` but **not** ```Vector a a a``` should be put in the function's type signature.
```Haskell
vplus :: (Num t) => Vector t -> Vector t -> Vector t
(Vector i j k) `vplus` (Vector l m n) = Vector (i+l) (j+m) (k+n)
```

## `newtype` is preferred to `data` under certain conditions
For a given type, if there is only 1 data constructor containing single field.  Key word ```newtype``` can be used instead of ```data```.
Example.
```Haskell
data Hm = Hm Int
```
is better written as
```Haskell
newtype Hm = Hm Int
```
This safeguards a mistake of assigning an ```Int``` value to the ```Hm``` type during compilation.
Before compiling to an executable, it generates the source code by removing the ```newtype```.
This avoids the runtime overhead of wrapping/unwrapping for the underlying value.

Type variable is also allowed in ```newtype```.  Example
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

Type class is used for function declaration, not for data type declaration.  
In some conditions, a type class instance can be defined only if another type class is satisfied.  
Example
```Haskell
instance (Eq m) => Eq (Maybe m) where
  -- implement Eq's required behaviour for the type Maybe m
```
i.e. to make ```Maybe m``` satisfy ```Eq```, ```m``` should satisfy ```Eq```.

## `Contravariant`
```
class Contravariant f where
    contramap :: (a -> b) -> f b -> f a
```

### Contravariant examples
In `r -> a`, `(r ->)` is a functor.  
In `a -> r`, `(-> r)` is a contravariant.

```
newtype ReverseFunction r a = ReverseFunction { func :: a -> r}

instance Contravariant (ReverseFunction r) where
    contramap f (ReverseFunction g) = ReverseFunction $ g . f
```

Therefore Json encoder is contravariant.

## `Generic` 101
Idea: all data types can be described by either sum type or product type or both.  Therefore there is a generic way to represent the data types s.t. the compiler can generate this common, low-level data representation automatically.  This idea is implemented using the type class as follows.

```
class Generic a where
    -- Convert a value to its generic representation
    from :: a -> Rep a

    -- Convert the generic representation back to the value
    to   :: Rep a -> a

```

To automatically generate generic representation for a data type, GHC extension `DeriveGeneric` is applied as follows.

```
{-# LANGUAGE DeriveGeneric #-}

import GHC.Generics (Generic)

data Person = Person { name :: String, age :: Int }
  deriving (Generic, Show)
```

Representing a data type in generic representation allow functions (e.g. serialization or deserialization) to be written in a way that they can work with any data type rather than being hardcoded.  `aeson` are great examples of functions that work on a data type's generic representation.

```
{-# LANGUAGE DeriveGeneric #-}

import GHC.Generics (Generic)
import Data.Aeson (ToJSON, FromJSON)

data Person = Person { name :: String, age :: Int }
  deriving (Generic, Show)

instance ToJSON Person
instance FromJSON Person
```

Note: The `Generic` type class is closely related to the isomorphism concept.

## `Exception` type class
* We can throw exceptions in pure function using `throw`.  But this is **highly not recommended**.  Instead, we should use `throwIO` whose return type is IO a.  The reason is pure code cannot catch exceptions.  Exceptions can only be handled within `IO`.  These handled exceptions are `Exception` instances.
* We can define Exception instances for application purposes.
* Even if we define the Exception instance, when we throw it by, say, `throwIO`, it will be wrapped automatically inside `SomeException` so that it can be caught by `catch` or `catchAll`.
* Consider an example, `handle handleHttpException ma` where `handleHttpException :: MonadCatch m => HttpException -> m a` and `HttpException` is `Exception`.  If another exception type is thrown during runtime, it won't be handled by `handleHttpException`, and be propagated as an unhandled exception, and depending on how the program is structured, it will be either
  * Caught by another handler up the chain or
  * Cause the program to terminate with an unhandled exception error.
* Given the note mentioned before, we can see that `Exception` does not represents all GHC runtime exceptions.  But GHC runtime generated exceptions are `Exception` instances.  Therefore these GHC runtime exceptions can be handled by the `Exception` framework.
* When to use Exception or not?  I think if the code is **pure**, we can define programmatic exceptions which is more readable.

## Automatic conversion - "secret" GHC behaviour
This is a mysterious feature if you are unfamiliar with the GHC behaviour.  The code still works w/o explicit conversion.  But it will simplify the code if you know this "secret" behaviour.  
Automatic conversion happens in a few occasions based on **type class** instances.   

### 1. `Num` 
If the type of an expression is `Num` instance, GHC will **automatically call** `fromInteger :: Num a => Integer -> a` for an integer literal.  E.g.
```
data Temp = Temp Double deriving (Num, Fractional)

paperBurning = 451 :: Temp
```

### 2. `Fractional`
Similarly, GHC will **automatically call** `fromRational :: Fractional a => Rational -> a` for a fractional literal.  E.g. 
```
absoluteZero = -273.15 :: Temp
```

### 3. `IsString` - requires`OverloadedStrings` extension
GHC will **automatically call** `fromString :: IsString a => String -> a` for a string literal.  E.g.
```
{-# LANGUAGE OverloadedStrings #-}

myText = "hello world" :: Text
```

### 4. `IsList` - requires`OverloadedLists` extension
GHC will **automatically call** `fromList :: IsList l => [Item l] -> l` for a list literal.  E.g.
```
{-# LANGUAGE OverloadedLists #-}

vector = [1,2,3,4,5] :: Vector Int
```

## Type variable scope
The following code has compilation error
```
unit :: UnitName u => Temp u -> String
unit _ = unitName (Proxy :: Proxy u)
```
The scope of `u` is within the function type signature only.  GHC doesn't know where the `u` of `Proxy u` comes from.  
The code is fixed by adding `forall`
```
unit :: forall u. UnitName u => Temp u -> String
unit _ = unitName (Proxy :: Proxy u)
```
`forall` broadens the scope of `u` from the type signature to the function implementation as well.  Since GHC 9.6.5 or maybe earlier, `ScopedTypeVariables` is not required in order to broaden the type variable scope.

## `Show` 101
When enter a value on GHCi, it tries to display the result using its `Show` by `print` implicitly.  
```
print :: Show a => a -> IO ()
print x = putStrLn (show x)
```
That explains why `show 'a'` gives `"'a'"` but `'a'` gives `'a'` on GHCi.

## Type families
Basic definition knowledge are assumed.  Observation is summarised:
* There are open and closed TFs.
* Associated TF means the TF is associated to a type class.  The TF instance is defined for the corresponding type class instance, therefore only open TF can be associated to a type class.
* Those TF not associated to a type class is called top-level TF.
* Unlike TF, a DF instance is not used to map from 1 existing type to another type.  It is used to define a new type.  This idea enables modular extensibility.  Therefore it doesn't make sense for an closed DF.
* TF is not limited to type-level programming.
* https://serokell.io/blog/type-families-haskell gave a bit in-depth

### Indexed type families
TFs are sometimes called indexed TFs where the parameter variable is the index.  E.g.
```Haskell
type family F a where
  F Int = Bool
  F Char = String
  F a = [a]
```
`a` is the index

### DF instance can define an ADT
```Haskell
data family Foo a
data instance Foo Int = FooInt Int | FooPair Int Int
data instance Foo Bool = FooTrue | FooFalse | FooMaybe Bool
```

## Pragmas
A pragma is a directive to the compiler.  It tells the compiler to enable a language extension that processes input in a way beyond what the standard provides for.

### GHC extensions learnt from "Haskell in depth"

* `OverloadedStrings` p. 14
* `DeriveAnyClass` p. 24
* `StandaloneDeriving` p. 31
* `NoImplicitPrelude` p. 105

### 1. `forall` and RankNTypes
Use `forall` to implement the idea of arbitrary-rank polymorphism.  I found `RankNTypes` is inferred w/o enabling the extension since GHC 9.6.5 or maybe earlier.

<u>Example 1</u>
```Haskell
processInts :: (a -> a) -> [Int] -> [Int]
processInts f = map f
```

Compilation failed.  The above code is indeed
```Haskell
processInts :: forall a. (a -> a) -> [Int] -> [Int]
processInts f = map f
```

It compiles if the scope of `forall` is changed as
```Haskell
processInts :: (forall a. a -> a) -> [Int] -> [Int]
processInts f = map f
```

Reasons: 
* The `forall` scope determines what function level `a` is universally quantified at.
* The function caller determines what `a` is.

In first case, `a` is universally quantified at `processInts`.  i.e. `a` is determined by the `processInts` caller but the `(a -> a)` passed by the caller may not satisify the implementation requirement.  
In second case, `a` is universally quantified at `f`.  i.e. `a` is determined by the `f` caller which in this case is the `processInts` implementation.  Therefore it compiles.  If the implementation doesn't call `f`, it will be somewhere else, not necessarily the `processInts` caller, determines `a`.

<u>Example 2</u>
```Haskell
applyToTuple :: ([a] -> Int) -> ([b], [c]) -> (Int, Int)
applyToTuple f (x, y) = (f x, f y)
```

Compilation failed due to same reason.  The code is indeed
```Haskell
applyToTuple :: forall a b c. ([a] -> Int) -> ([b], [c]) -> (Int, Int)
applyToTuple f (x, y) = (f x, f y)
```

It compiles if the scope of `forall` is changed as
```Haskell
applyToTuple :: (forall a. [a] -> Int) -> ([b], [c]) -> (Int, Int)
applyToTuple f (x, y) = (f x, f y)
```

Note for the modified `applyToTuple`:
* A variable is **universally quantified** when the consumer of the expression it appears in can choose what it will be.  E.g. `b` and `c`.  Because `applyToTuple` users can choose what the types `b` and `c` are.
* A variable is **existentially quantified** when the consumer of the expression it appears in have to deal with the fact that the choice was made for him.  E.g. `a`.  Because it has to deal wih any type given to it.
* Once entered inside `applyToTuple`, `b` and `c` are then **existentially quantified** because this function user has already chosen the types from calling.

Reference:
* https://markkarpov.com/post/existential-quantification.html
* http://sleepomeno.github.io/blog/2014/02/12/Explaining-Haskell-RankNTypes-for-all/

<u>Example 3</u>
```Haskell
newtype NumModifier = NumModifier {
  run :: a -> a
}
```
It complains `a` is not in scope.  This is defining a type, not a function.  The fix is either adding `a` on the type level or using `forall` as follows.  

```Haskell
newtype NumModifier = NumModifier {
  run :: forall a. a -> a
}
```
`a` is not applicable for the data type `NumModifier` but only within the field `run`.

```Haskell
data NumModifier = NumModifier {
   run1 :: forall a. a -> a
 , run2 :: forall a. Num a => a -> a
}
```
`a` in `run1` and `a` in `run2` are completely independent.

### 2. `forall` and `ExistentialQuantification`
In this case `forall` is used on defining a type.  E.g. 
```
data SomeException = forall e . Exception e => SomeException e
```
Usually it's an error because any type variable appearing on the right must also appear on the left.  It can be fixed by
```
data SomeException e = SomeException e
```
However if `e` always belongs to a type class, say, `Exception`, then all functions using `SomeException` should be declared in this way
```
func :: Exception e => SomeException e
```
Using existential types solves this problem.
```
{-# LANGUAGE ExistentialQuantification #-}
data SomeException = forall e. Exception e => SomeException e
func :: SomeException
```

#### Why existentials?
Existentials are always about throwing type information away. Why would we want to do that? The answer is: sometimes we want to work with types that we don’t know at compile time. The types typically depend on the state of external world: they could depend on user’s input, on contents of a file we’re trying to parse, etc.

Reference:
* https://markkarpov.com/post/existential-quantification.html
* https://wiki.haskell.org/Existential_type

### 3. `GeneralizedNewtypeDeriving` - make a type class derivable
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

Another sample usage of `GeneralizedNewtypeDeriving`
```
{-# LANGUAGE GeneralizedNewtypeDeriving  #-}
newtype MyApp a = MyApp {
      runApp :: ReaderT WebAPIAuth IO a
    } deriving (Functor, Applicative, Monad, MonadIO,
                MonadThrow, MonadCatch, MonadMask,
                MonadReader WebAPIAuth)
```

### 4. `InstanceSigs` - allows type signature in defining the type class instance
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

### 5. Data declaration with constraint
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

### 6. Generalized Algebraic Data Types (GADTs)
A parameterised ADT enforces all its constructors to have the **same parameterised type**.  GADT lifts this restriction to allow each of its constructor to specify its own parameterised type.  

```Haskell
data Expr a where
  I   :: Int  -> Expr Int
  B   :: Bool -> Expr Bool
  Add :: Expr Int -> Expr Int -> Expr Int
  Mul :: Expr Int -> Expr Int -> Expr Int
  Eq  :: Eq a => Expr a -> Expr a -> Expr Bool
```

Without GADT, `I 1` and `B True` produce the same type s.t. `Eq (I 1) (B True)` compiles.  But GADT enables compilation error.  

* For more information, refer to https://en.wikibooks.org/wiki/Haskell/GADT.  
* Since GHC 9.6.5 or maybe earlier, `GADTs` is not required to be enabled to define a GADT.

### 7. `RecordWildCards` for `{..}` on data constructor having record syntax
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

### 8. `ViewPatterns`
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
* Parentheses must be needed

### 9. `FunctionalDependencies`
It is used on multiple parameter type classes.  
E.g. `class Monad m => MonadError e m | m -> e where`, `m` determines `e`, it means for a given `m` value, it should be related to at most one `e` value in the `MonadError` type class.  E.g. when `instance MonadError IOException IO where` is defined, `instance MonadError String IO where` will get compilation error.

Another easy example comes from https://www.schoolofhaskell.com/school/to-infinity-and-beyond/pick-of-the-week/type-families-and-pokemon.

Suppose there are 3 types of pokemons - `Fire`, `Water`, `Grass` and 3 types of pokemon moves - `FireMove`, `WaterMove`, `GrassMove` such that 
```
data Fire = Charmander | Charmeleon | Charizard
data Water = Squirtle | Wartortle | Blastoise
data Grass = Bulbasaur | Ivysaur | Venusaur

data FireMove = Ember | FlameThrower | FireBlast
data WaterMove = Bubble | WaterGun
data GrassMove = VineWhip
```

At this stage, we can tell `Fire` is related to `FireMove` only and so on.  Suppose we want to define a function to get a move from a pokemon, we can define a type class as follows:
```
{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies #-}

class Pokemon pokemon move | pokemon -> move where
  pickMove :: pokemon -> move

instance Pokemon Fire FireMove where
    pickMove Charmander = Ember
    pickMove Charmeleon = FlameThrower
    pickMove Charizard = FireBlast

instance Pokemon Water WaterMove where
    pickMove Squirtle = Bubble
    pickMove _ = WaterGun

instance Pokemon Grass GrassMove where
    pickMove = const VineWhip
```
Meaningless relationship such as `instance Pokemon Fire WaterMove` won't pass the compilation.

### 10. `TypeFamilies`
Besides `FunctionalDependencies`, `TypeFamilies` is an alternative to type check the pokemon example.
```
{-# LANGUAGE TypeFamilies #-}
class Pokemon pokemon where
  data Move pokemon :: *
  pickMove :: pokemon -> Move pokemon

instance Pokemon Fire where
  data Move Fire = Ember | FlameThrower | FireBlast
  pickMove Charmander = Ember
  pickMove Charmeleon = FlameThrower
  pickMove Charizard = FireBlast

instance Pokemon Water where
  data Move Water = Bubble | WaterGun
  pickMove Squirtle = Bubble
  pickMove _ = WaterGun

instance Pokemon Grass where
  data Move Grass = VineWhip
  pickMove = const VineWhip
```

Comparing with `FunctionalDependencies`, `TypeFamilies` seems a bit less trivial in the beginning, however it has the advantages of less type parameter.

References: 
* https://www.schoolofhaskell.com/school/to-infinity-and-beyond/pick-of-the-week/type-families-and-pokemon.
* http://www.mchaver.com/posts/2017-06-21-type-families.html

### 11. `OverloadedStrings`
It has been explained in the section of automatic conversion.

### 12. `OverloadedLists`
It has been explained in the section of automatic conversion.

### 13. `DeriveGeneric`
It has been explained in the section of data type generic representation.

### 14. `DeriveAnyClass`
Get back to the `aeson` example again.
```
{-# LANGUAGE DeriveGeneric #-}

import GHC.Generics (Generic)
import Data.Aeson (ToJSON, FromJSON)

data Person = Person { name :: String, age :: Int }
  deriving (Generic, Show)

instance ToJSON Person
instance FromJSON Person
```

We can remove the last 2 lines by deriving `ToJSON` and `FromJSON` as follows.
```
{-# LANGUAGE DeriveAnyClass #-} 
{-# LANGUAGE DeriveGeneric #-}

import GHC.Generics (Generic)
import Data.Aeson (FromJSON, ToJSON) 

data Person = Person { name :: String , age :: Int } deriving (Show, Generic, FromJSON, ToJSON)
```

Note: `Show`, `Eq`, `Ord` have built-in support for deriving, `DeriveAnyClass` is not required for these type classes.

### 15. `AllowAmbiguousTypes`
The following code doesn't compile.
```
class UnitName u where
  unitName :: String
```
The reason the function `unitName` doesn't have an argument depending on the type variable `u` abstracted by the type class.  Enabling `AllowAmbiguousTypes` makes it compile.  However, the function still doesn't know the concrete type (or kind in general) when it's called.

### 16. `TypeApplications`
It addresses the problem mentioned in the previous section.  Even the code compiles, `unitName` doesn't have an argument depending on the type variable targetted by the type class, we can't call it.  Enabling `TypeApplications` solves the problem.  Suppose there is a concrete type, say, `F` and its `UnitName` instance, it can be called as
```
unitName @F
```

The `UnitName` example illustrates how to use `AllowAmbiguousTypes` and `TypeApplications` to carry type information w/o requiring a value of that type.  An alternative solution is using `Data.Proxy` as
```
import Data.Proxy

class UnitName u where
  unitName :: Proxy u -> String
```
Suppose there is a type `F` and its `UnitName` instance, the function can be called as
```
unitName (Proxy :: Proxy F)
```  

**Note on using `@` when `TypeApplications` enabled**  
* If the function is polymorphic, we can use `@`.  This function is not necessarily from a type class such as `unitName`
* `@` is usually placed right after the function name
  * `read @Int "42"`
  * `map @Int show [1, 2, 3]` - although it's not needed given the type inference
* What if there are multiple type variables such as `fmap :: Functor f => (a -> b) -> f a -> f`?  The order will be `fmap @[] @Int @String show [1,2,3]`.

### 17. `ConstraintKinds`
`type Show' a = Show a` gets compilation error.  Fix it by enabling `ConstraintKinds`.  

Usage:
* `type Showable a b = (Show a, Show b)`  
  `printPair :: Showable a b => a -> b -> IO ()`
* `type UnescapingShow t = (ToUnescaping t, Show (ToUnescapingTF t))`  
  `ushow :: UnescapingShow t => t -> String`  

This avoids repeating multiple type class requirement across function declaration.