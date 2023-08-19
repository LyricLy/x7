module State where

import Control.Lens hiding (List)
import Control.Monad
import Control.Monad.State
import Control.Monad.Except
import Data.Ratio
import Data.Char
import Data.Data
import Data.Data.Lens
import Data.Foldable
import Data.List
import Data.List.NonEmpty (NonEmpty(..))
import Data.Map (Map)
import qualified Data.Map as M
import Data.Void
import Data.Sequence (Seq)
import Error.Diagnose

showRat :: Rational -> String
showRat n
  | b == 0 = whole
  | otherwise = whole ++ '.' : fst (go b [])
  where
    split = flip divMod (denominator n)
    (a, b) = split . abs $ numerator n
    whole = (if n < 0 then ('-':) else id) (show a)
    go 0 _ = ("", Nothing)
    go v p =
      let (q, r) = split (v * 10)
          p' = v : p
          (x, k)
            | r `elem` p' = ("", Just r)
            | otherwise = go r (v : p)
          x' = intToDigit (fromIntegral q) : x
          x''
            | k == Just v = "(" ++ x' ++ ")"
            | otherwise = x'
      in (x'', k)

data ValueOf a
  = Rat Rational
  | Box (ValueOf a)
  | Pair (ValueOf a, ValueOf a)
  | List { _list :: Seq (ValueOf a) }
  | Focused a
  deriving (Data, Functor)
makeLenses ''ValueOf
makePrisms ''ValueOf

_Integer :: Prism' Value Integer
_Integer = _Rat . prism' fromIntegral \x -> if denominator x == 1 then Just (numerator x) else Nothing 

type Value = ValueOf Void
type FocusedValue = ValueOf Value

instance Plated FocusedValue

data Depth = Top | Static | Single | Deep deriving (Show, Eq, Ord)
data View = View {_val :: FocusedValue, _depth :: (Depth, Bool)}
makeLenses ''View
type Group = NonEmpty View
type Stack = [Group]

data Place = Place {_stack :: Stack, _vars :: Map Char View}
makeLenses ''Place

instance Show a => Show (ValueOf a) where
  show (Rat v) = showRat v
  show (Box v) = '&' : show v
  show (Pair v) = show v
  show (List v) = show (toList v)
  show (Focused v) = "<" ++ show v ++ ">"

instance Show View where
  show v = case v^.val of
    Focused x -> show x
    x -> show x

instance Show Place where
  show p = unwords . map showGroup $ reverse (p^.stack)
    where showGroup xs = intercalate "$" . map show . reverse $ toList xs

data Raise = Mask Raise | RaiseWithContext Position Place String | Raise String
makePrisms ''Raise

reportRaise :: Raise -> Report String
reportRaise (Mask (reportRaise -> (Err f0 msg f1 f2))) = Err f0 (msg ++ "(masked)") f1 f2
reportRaise (RaiseWithContext pos st t) = Err Nothing "instruction raised" [(pos, This t)] [Note note]
  where note = let s = show st in if null s then "stack is empty" else "stack contents: " ++ s
reportRaise _ = error "reportRaise: got plain Raise"

type X7 = StateT Place (Except Raise)

raise :: String -> X7 a
raise = throwError . Raise

typeError :: X7 a
typeError = raise "type error"

deSpan :: Position -> X7 () -> X7 ()
deSpan p m = do
  s <- get
  withError (id & outside _Raise .~ RaiseWithContext p s) m

getVar :: Char -> X7 View
getVar c = use (vars . at c) >>= maybe (raise $ "variable '" ++ c : "' not defined") pure

setVar :: Char -> View -> X7 ()
setVar c v = vars . at c ?= v 

popGroup :: X7 Group
popGroup = join $ stack %%= \case
  [] -> (raise "stack underflow", [])
  x:xs -> (pure x, xs)

pushGroup :: Group -> X7 ()
pushGroup g = stack %= (g:)

ofValue :: Value -> View
ofValue x = View (Focused x) (Top, False)

popView :: X7 View
popView = popGroup >>= \(x:|xs) -> x <$ mapM_ pushView xs

pushView :: View -> X7 ()
pushView = pushGroup . pure

focused :: Traversal' View Value
focused = val . biplate

unfocus :: View -> View
unfocus = ofValue . unfocusValue . view val
  where
    unfocusValue = fmap undefined . transform \case
      Focused x -> fmap absurd x
      x -> x

focus :: Monad m => (Value -> m FocusedValue) -> View -> m View
focus f = val %%~ transformM \case
  Focused x -> f x
  x -> pure x

deepen :: Depth -> View -> View
deepen l = depth %~ \case
  (_, True) -> (Deep, True)
  (v, _) -> (max l v, False)

focus' :: Depth -> (Value -> X7 FocusedValue) -> View -> X7 View
focus' d f v = deepen d <$> focus f v

flatten :: View -> Maybe View
flatten = depth._2 .~ True <&> focus \case
  List v -> Just (List (fmap Focused v))
  _ -> Nothing

type Drill = View -> X7 View

drillFrom :: Depth -> Drill
drillFrom d v = if v^.depth._1 <= d then maybe (raise "type error (can't drill)") pure (flatten v) else pure v

drillAtom :: Drill
drillAtom x@(flatten -> Just v)
  | has focused x = drillAtom v
drillAtom x = pure x

op :: Drill -> (Value -> X7 Value) -> X7 ()
op drill f = popView >>= drill >>= focused %%~ f >>= pushView . unfocus

through :: Traversal' Value a -> Review Value b -> (a -> b) -> (Value -> X7 Value)
through a b f x = case f <$> x^?a of
  Just r -> pure $ b # r
  Nothing -> typeError

through' :: Prism' Value a -> (a -> a) -> (Value -> X7 Value)
through' p f = through p p f

op2 :: Drill -> (Value -> Value -> X7 Value) -> X7 ()
op2 drill f = do
  let v = popView >>= drill
  y <- v
  x <- v
  let foc = y^..focused
  when (null foc) (raise "rhs has nothing in focus")
  r <- x & partsOf focused %%~ flip (zipWithM f) (cycle foc)
  pushView (unfocus r)
  pure ()

through2 :: Traversal' Value a -> Traversal' Value b -> Review Value c -> (a -> b -> c) -> (Value -> Value -> X7 Value)
through2 a b c f x y =
  case f <$> x^?a <*> y^?b of
    Just r -> pure $ c # r
    Nothing -> typeError

through2' :: Prism' Value a -> (a -> a -> a) -> (Value -> Value -> X7 Value)
through2' p f = through2 p p p f

runX7 :: X7 () -> Either Raise Place
runX7 x = runExcept (execStateT x (Place [] M.empty))
