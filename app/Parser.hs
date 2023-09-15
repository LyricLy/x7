module Parser where

import Control.Applicative (liftA2)
import Control.Monad
import Control.Monad.Reader
import Control.Lens hiding (op, List)
import Data.Bool
import Data.Char
import Data.List.NonEmpty (NonEmpty, nub)
import Data.Maybe
import Data.Sequence (fromList)
import Data.Validation
import Data.Void
import Error.Diagnose
import Text.Megaparsec
import Text.Megaparsec.Char
import State

type Parser = Parsec Void String

data ElaborationError = FuncNotDefined Position Int Int

instance Eq ElaborationError where
  (FuncNotDefined _ _ x) == (FuncNotDefined _ _ y) = x == y

type Inst = Position -> Inst'
-- an Inst that already has the position data it needs
type Inst' = ReaderT [X7 ()] (Validation (NonEmpty ElaborationError)) (X7 ())

attachSpan :: Position -> Inst -> Inst'
attachSpan p i = deSpan p <$> i p

callInst :: Int -> Inst
callInst n pos = ReaderT \fns ->
  case fns ^? ix (n-1) of
    Just x -> Success x
    Nothing -> Failure [FuncNotDefined pos (length fns) n]

sequenceInst :: [Inst'] -> Inst'
sequenceInst = mapReaderT (_Failure %~ nub) . fmap sequence_ . sequenceA

nonDigitChar :: Parser Char
nonDigitChar = satisfy (not . isDigit) <?> "non-digit"

nonZeroDec :: Parser Int
nonZeroDec = read <$> liftA2 (:) (satisfy (liftA2 (&&) (/='0') isDigit)) (many digitChar) <?> "nonzero number"

intLit :: Parser Inst
intLit = const . pure . pushValue . Rat . fromIntegral <$> (0 <$ char '0' <|> nonZeroDec)

varGet :: Parser Inst
varGet = char ';' >> const . pure . (>>= pushView) . getVar <$> nonDigitChar

varSet :: Parser Inst
varSet = char ':' >> const . pure . (popView >>=)  . setVar <$> nonDigitChar

funCall :: Parser Inst
funCall = char ';' >> callInst <$> nonZeroDec

lastBlock :: Parser Inst'
lastBlock = series <* optional (char '`')

-- look into making the } optional here; any fun opportunities or is it just confusing?
initBlock :: Parser Inst'
initBlock = series <* char '}'

o :: Char -> X7 () -> Parser Inst
o c r = const (pure r) <$ char c

o1 :: Char -> (X7 () -> X7 ()) -> Parser Inst
o1 c r = const . fmap r <$> (char c *> lastBlock)

o2 :: Char -> (X7 () -> X7 () -> X7 ()) -> Parser Inst
o2 c r = (const .) . liftA2 r <$> (char c *> initBlock) <*> lastBlock

inst :: Parser Inst
inst = intLit <|> varSet <|> try varGet <|> funCall
  <|> o '+' (op2 drillAtom $ through2' _Rat \x y -> pure $ x + y)
  <|> o '-' (op2 drillAtom $ through2' _Rat \x y -> pure $ x - y)
  <|> o '*' (op2 drillAtom $ through2' _Rat \x y -> pure $ x * y)
  <|> o 'Q' (op2 drillAtom $ through2' _Integer \x y ->
    if y == 0 then raise "division by zero" else pure $ x `div` y)
  <|> o 'R' (op2 drillAtom $ through2' _Integer \x y ->
    if y == 0 then raise "modulo by zero" else pure $ x `rem` y)
  <|> o 'D' (op2 drillAtom $ through2' _Rat \x y ->
    if y == 0 then raise "division by zero" else pure $ x / y)
  <|> o 'N' (op drillAtom $ through' _Rat $ pure . negate)
  <|> o 'J' (op drillAtom $ through' _Rat $ pure . fromIntegral @Integer . floor)
  <|> o 'K' (op drillAtom $ through' _Rat $ pure . fromIntegral @Integer . ceiling)
  <|> o '<' (comparison (<))
  <|> o 'G' (comparison (>=))
  <|> o '=' (comparison (==))
  <|> o '/' (comparison (/=))
  <|> o '>' (comparison (>))
  <|> o 'L' (comparison (<=))
  <|> o ',' (op2 pure \x y -> pure $ Pair (x, y))
  <|> o '[' (pushValue $ List mempty)
  <|> o '.' (op2 pure \x y -> maybe typeCompatError pure (dot x y))
  <|> o 'i' (op drillAtom $ through _PosInt id \n -> pure . List $ fmap (Rat . fromIntegral) [0..n-1])
  <|> o 'h' (opTic drillAtom Static $ through _Pair _Pair \(x, y) -> pure (Focused x, unFocus y))
  <|> o 't' (opTic drillAtom Static $ through _Pair _Pair \(x, y) -> pure (unFocus x, Focused y))
  <|> o 'j' (popView >>= flatten' >>= pushView)
  <|> o 'n' do
    i <- popView >>= drillAtom
    l <- popView
    l' <- drillSelMany l
    i' <- mapM (through _PosInt id pure) (i^..focused)
    case indexView notElem i' l' of
      Nothing -> indexError
      Just v ->
        let d = max (resolveDepth l) if resolveDepth i > Single then Deep else Single
        in pushView $ setDepth d v & onePerList .~ True
  <|> o 'u' do
    i <- popView >>= drillAtom
    l <- popView >>= drillSelMany
    i' <- mapM (through _PosInt id pure) (i^..focused)
    case indexView elem i' l of
      Nothing -> indexError
      Just v -> pushView $ deepen Deep v
  <|> o1 'w' (\b -> opTic pure Top \x -> bool (Focused x) (unFocus x) <$> raises (pushValue x >> b))
  <|> o '@' do
    l <- popView >>= drillFrom Top
    case l^..focused of
      [x] -> pushValue x
      xs -> addNote "'@' requires exactly one value to be focused" . raise $ show (length xs) ++ " values are focused"
  <|> o ']' (popView >>= drillEnlist >>= pushValue . List . fromList . toListOf focused)
  <?> "an instruction"

curlyBraces :: Parser Inst'
curlyBraces = pure (pure ()) <$ char '}' <|> char '{' *> series <* char '}'

inst' :: Parser Inst'
inst' = curlyBraces <|> do
  SourcePos f row1 col1 <- getSourcePos
  i <- inst
  SourcePos _ row2 col2 <- getSourcePos
  pure $ attachSpan (Position (unPos row1, unPos col1) (unPos row2, unPos col2) f) i

sc :: Parser ()
sc = hidden hspace

series :: Parser Inst'
series = sc >> sequenceInst <$> many (inst' <* sc)

x7 :: Parser [Inst']
x7 = [] <$ hidden eof <|> liftA2 (:) (series <* (void newline <|> eof <?> "end of line")) x7

errorToReport :: ElaborationError -> Report String
errorToReport = \case
  FuncNotDefined pos l n -> let
    msg
      | l == 1 = "there is only one line in the file"
      | otherwise = "there are only " ++ show l ++ " lines in the file"
    in Err Nothing ("function " ++ show n ++ " is not defined") [(pos, This msg)] []

elaborate :: [Inst'] -> Either (NonEmpty (Report String)) (X7 ())
elaborate fs =
  let fs' = fs <&> flip runReaderT (map (validation undefined id) fs')
  in case sequenceA fs' of
    Success xs -> Right $ fromMaybe (pure ()) $ xs ^? _last
    Failure rs -> Left $ fmap errorToReport rs
