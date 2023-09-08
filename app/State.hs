module State where

import Control.Lens hiding (List)
import Control.Monad
import Control.Monad.State
import Control.Monad.Except
import Data.Char
import Data.Data
import Data.Data.Lens
import Data.Foldable
import Data.Functor
import Data.List
import Data.List.NonEmpty (NonEmpty(..))
import Data.Map (Map)
import qualified Data.Map as M
import Data.Maybe
import Data.Ratio
import Data.Void
import Data.Sequence (Seq, (!?), (><))
import Error.Diagnose
import Witherable.Lens
import qualified Witherable.Lens.Withering as W

showRat :: Rational -> String
showRat n
  | b == 0 = whole
  | Just (r, _) <- go b [] = whole ++ '.' : r
  | otherwise = whole' ++ show b ++ '/' : show (denominator n)
  where
    split = flip divMod (denominator n)
    (a, b) = split . abs $ numerator n
    theMinus
      | n < 0 = "-"
      | otherwise = ""
    whole = theMinus ++ show a
    whole'
      | a == 0 = theMinus
      | n < 0 = whole ++ "-"
      | otherwise = whole ++ "+"
    go 0 _ = Just ("", Nothing)
    go v p
      | length p == 20 = Nothing
      | otherwise =
        let (q, r) = split (v * 10)
            p' = v : p
        in do
          (x, k) <- if r `elem` p' then Just ("", Just r) else go r p'
          let x' = intToDigit (fromIntegral q) : x
          let x'' = if k == Just v then "(" ++ x' ++ ")" else x'
          pure (x'', k)

data ValueOf a
  = Rat Rational
  | Box (ValueOf a)
  | Pair (ValueOf a, ValueOf a)
  | List (Seq (ValueOf a))
  | Focused a
  deriving (Data, Functor, Eq, Ord)
makeLenses ''ValueOf
makePrisms ''ValueOf

instance Show a => Show (ValueOf a) where
  show (Rat v) = showRat v
  show (Box v) = '#' : show v
  show (Pair v) = show v
  show (List v) = show (toList v)
  show (Focused v) = "<" ++ show v ++ ">"

type Value = ValueOf Void
type FocusedValue = ValueOf Value

sameType :: Value -> Value -> Bool
sameType (Rat _) (Rat _) = True
sameType (Box _) (Box _) = True
sameType (Pair (x1, x2)) (Pair (y1, y2)) = sameType x1 y1 && sameType x2 y2
sameType (List xs) (List ys) = fromMaybe True . listToMaybe $ zipWith sameType (toList xs) (toList ys)
sameType _ _ = False

maybeSameType :: Maybe Value -> Maybe Value -> Bool
maybeSameType = (fromMaybe True .) . liftM2 sameType

dot :: Value -> Value -> Maybe Value
dot (List x) (List y)
  | maybeSameType (x !? 0) (y !? 0) = Just $ List (x >< y)
dot (List x) y
  | maybeSameType (x !? 0) (Just y) = Just $ List (x |> y)
dot x (List y)
  | maybeSameType (Just x) (y !? 0) = Just $ List (x <| y)
dot x y
  | sameType x y = Just $ List [x, y]
dot _ _ = Nothing

_Integer :: Prism' Value Integer
_Integer = _Rat . prism' fromIntegral \x -> numerator x <$ guard (denominator x == 1)
_PosInt :: Prism' Value Int
_PosInt = _Integer . prism' fromIntegral \x -> fromIntegral x <$ guard (x >= 0)
instance Plated FocusedValue

data Depth = Top | Static | Single | Deep deriving (Eq, Ord)

instance Show Depth where
  show Top = "top"
  show Static = "static"
  show Single = "single"
  show Deep = "deep"

data View = View {_val :: FocusedValue, _depth :: Depth, _flattened :: Bool, _onePerList :: Bool}
makeLenses ''View
type Group = NonEmpty View
type Stack = [Group]

data Place = Place {_stack :: Stack, _vars :: Map Char View}
makeLenses ''Place

instance Show View where
  show = (show & outside _Focused .~ show) . view val

instance Show Place where
  show p = unwords . map showGroup $ reverse (p^.stack)
    where showGroup xs = intercalate "&" . map show . reverse $ toList xs

data RaiseData = RaiseData {_text :: String, _notes :: [String]}
makeLenses ''RaiseData
data Raise = Mask Raise | RaiseWithContext Position Place RaiseData | Raise RaiseData
makePrisms ''Raise

reportRaise :: Raise -> Report String
reportRaise r = go r 0
  where
    go :: Raise -> Int -> Report String
    go (Mask t) n = go t (n+1)
    go (RaiseWithContext pos st (RaiseData t ns)) n =
      let s = show st
          note
            | null s = "stack is empty"
            | otherwise = "stack contents: " ++ s
          masked
            | n == 0 = ""
            | n == 1 = " (masked)"
            | otherwise = " (masked " ++ show n ++ " times)"
      in Err Nothing ("instruction raised" ++ masked) [(pos, This t)] (map Note (note : ns))
    go _ _ = error "reportRaise: got plain Raise"

type X7 = ExceptT Raise (State Place)

raise :: String -> X7 a
raise s = throwError $ Raise (RaiseData s [])

addNote :: String -> X7 a -> X7 a
addNote s = withError (_Raise.notes %~ (s:))

deSpan :: Position -> X7 () -> X7 ()
deSpan p m = do
  s <- get
  withError (id & outside _Raise .~ RaiseWithContext p s) m

raises :: X7 () -> X7 Bool
raises b = do
  s <- get
  (False <$ b) `catchError` \case
    Mask e -> throwError e
    _ -> True <$ put s

typeError :: X7 a
typeError = raise "argument has unexpected type"

typeCompatError :: X7 a
typeCompatError = raise "arguments are of incompatible types"

indexError :: X7 a
indexError = raise "index out of bounds"

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
ofValue x = View (Focused x) Top False True

popView :: X7 View
popView = popGroup >>= \(x:|xs) -> x <$ mapM_ pushView xs

pushView :: View -> X7 ()
pushView = pushGroup . pure

pushValue :: Value -> X7 ()
pushValue = pushView . ofValue

focusedV :: Traversal' FocusedValue Value
focusedV = biplate

focused :: Traversal' View Value
focused = val . focusedV

unFocus :: Value -> FocusedValue
unFocus = fmap absurd

rmFocus :: FocusedValue -> FocusedValue
rmFocus = transform $ id & outside _Focused .~ unFocus

defocus :: View -> View
defocus = ofValue . fmap undefined . rmFocus . view val

focus :: Monad m => (Value -> m FocusedValue) -> View -> m View
focus f = val %%~ transformM \case
  Focused x -> f x
  x -> pure x

setDepth :: Depth -> View -> View
setDepth = (set flattened False .) . set depth

resolveDepth :: View -> Depth
resolveDepth v
  | v^.flattened = Deep
  | otherwise = v^.depth

deepen :: Depth -> View -> View
deepen l = join $ setDepth . max l . resolveDepth

focus' :: Depth -> (Value -> X7 FocusedValue) -> View -> X7 View
focus' d f v = deepen d <$> focus f v

opTic :: Drill -> Depth -> (Value -> X7 FocusedValue) -> X7 ()
opTic drill d f = popView >>= drill >>= focus' d f >>= pushView

flatten :: View -> Maybe View
flatten = onePerList .~ False <&> flattened .~ True <&> focus \case
  List v -> Just (List (fmap Focused v))
  _ -> Nothing

flatten' :: View -> X7 View
flatten' = maybe (raise "focus is not list (can't drill)") pure . flatten

type Drill = View -> X7 View

drillFrom :: Depth -> Drill
drillFrom d v
  | v^.depth <= d = addNote ("instruction drills from " ++ show d ++ " and argument is at " ++ show (v^.depth)) $ flatten' v
  | otherwise = pure v

drillSelMany :: Drill
drillSelMany l
  | l^.onePerList = addNote "instruction drills when 'j' has not been used (implicitly or explicitly) since the last time 'n' was used" $ flatten' l
  | otherwise = pure l

drillAtom :: Drill
drillAtom x@(flatten -> Just v)
  | has focused x = drillAtom v
drillAtom x = pure x

op :: Drill -> (Value -> X7 Value) -> X7 ()
op drill f = popView >>= drill >>= focused %%~ f >>= pushView . defocus

through :: Traversal' Value a -> Review c b -> (a -> X7 b) -> (Value -> X7 c)
through a b f x = case f <$> x^?a of
  Just r -> review b <$> r
  Nothing -> typeError

through' :: Prism' Value a -> (a -> X7 a) -> (Value -> X7 Value)
through' p f = through p p f

type Traversoid f a b = LensLike' f View [Value] -> ([Value] -> X7 [a]) -> View -> X7 b

op2' :: Functor f => Traversoid f a b -> Drill -> (Value -> Value -> X7 a) -> X7 b
op2' z drill f = do
  let v = popView >>= drill
  y <- v
  x <- v
  let foc = y^..focused
  when (null foc) (raise "rhs has nothing in focus")
  x & z (partsOf focused) (flip (zipWithM f) (cycle foc))

op2 :: Drill -> (Value -> Value -> X7 Value) -> X7 ()
op2 d f = op2' id d f >>= pushView . defocus

op2_ :: Drill -> (Value -> Value -> X7 a) -> X7 ()
op2_ = op2' traverseOf_

through2 :: Traversal' Value a -> Traversal' Value b -> Review Value c -> (a -> b -> X7 c) -> (Value -> Value -> X7 Value)
through2 a b c f x y =
  case f <$> x^?a <*> y^?b of
    Just r -> review c <$> r
    Nothing -> typeError

through2' :: Prism' Value a -> (a -> a -> X7 a) -> (Value -> Value -> X7 Value)
through2' p f = through2 p p p f

comparison :: (Value -> Value -> Bool) -> X7 ()
comparison f = op2_ pure \x y ->
  if not $ sameType x y then typeCompatError
  else when (not $ f x y) $ raise "comparison failed"

-- wow this sucks!
flattenCount :: Seq FocusedValue -> Int
flattenCount = maybe 0 (1+) . minimumOf (traverse . failing (biplate . to flattenCount) (like (-1)))

selectionLists :: Traversal' (Seq FocusedValue) (Seq FocusedValue)
selectionLists f v = go (flattenCount v) f v
  where
    go 0 = id
    go n = traverse . biplate . go (n-1)

hunt :: Traversal' View (Seq FocusedValue)
hunt = val . _List . selectionLists

type Withering s t a b = forall f. Applicative f => (a -> W.Withering f b) -> s -> f t
type Withering' s a = Withering s s a a

selectedOf :: Traversal' FocusedValue FocusedValue
selectedOf = filtered (has focusedV)

selected :: Withering' View Value
selected = hunt . withered . selectedOf . focusedV

indexView :: (Int -> [Int] -> Bool) -> [Int] -> View -> Maybe View
indexView e is v =
  (v & hunt . elementsOf (traverse.selectedOf) (`e` is) %~ rmFocus)
  <$ mapM_ (\i -> v ^? hunt . elementOf (traverse.selectedOf) i) is

runX7 :: X7 () -> Either Raise Place
runX7 = uncurry ($>) . flip runState (Place [] M.empty) . runExceptT
