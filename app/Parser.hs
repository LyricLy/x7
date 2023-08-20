module Parser where

import Control.Applicative (liftA2)
import Control.Monad
import Control.Lens hiding (op, List)
import Data.Char
import Data.Either
import Data.List
import Data.Maybe
import Data.Void
import Error.Diagnose
import Text.Megaparsec
import Text.Megaparsec.Char
import State

type Parser = Parsec Void String

data Inst = Pure (X7 ()) | Call Int
data SpanInst = SpanInst Inst Position

nonDigitChar :: Parser Char
nonDigitChar = satisfy (not . isDigit) <?> "non-digit"

nonZeroDec :: Parser Int
nonZeroDec = read <$> liftA2 (:) (satisfy (liftA2 (&&) (/='0') isDigit)) (many digitChar) <?> "nonzero number"

intLit :: Parser Inst
intLit = Pure . pushView . ofValue . Rat . fromIntegral <$> (0 <$ char '0' <|> nonZeroDec)

varGet :: Parser Inst
varGet = char ';' >> Pure . (>>= pushView) . getVar <$> nonDigitChar

varSet :: Parser Inst
varSet = char ':' >> Pure . (popView >>=)  . setVar <$> nonDigitChar

funCall :: Parser Inst
funCall = char ';' >> Call <$> nonZeroDec

o :: Char -> X7 () -> Parser Inst
o c r = Pure r <$ char c

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
  <|> o '[' (pushView . ofValue $ List mempty)
  <|> o '.' (op2 pure \x y -> maybe typeError pure (dot x y))
  <|> o ']' (op pure $ pure . List . pure)
  <|> o 'i' (op drillAtom $ through _PosInt id \n -> pure . List $ fmap (Rat . fromIntegral) [0..n-1])
  <|> o 'h' (opTic drillAtom Static $ through _Pair _Pair $ \(x, y) -> pure (Focused x, noFocus y))
  <|> o 't' (opTic drillAtom Static $ through _Pair _Pair $ \(x, y) -> pure (Focused x, noFocus y))
  <|> o 'j' (popView >>= maybe typeError pure . flatten >>= pushView)
  <?> "an instruction"

sc :: Parser ()
sc = hidden hspace

func :: Parser [SpanInst]
func = sc >> many do
  SourcePos f row1 col1 <- getSourcePos
  i <- inst
  SourcePos _ row2 col2 <- getSourcePos
  sc
  pure . SpanInst i $ Position (unPos row1, unPos col1) (unPos row2, unPos col2) f

x7 :: Parser [[SpanInst]]
x7 = [] <$ eof <|> liftA2 (:) (func <* (void newline <|> eof)) x7

data ElaborationError = FuncNotDefined Position Int Int

instance Eq ElaborationError where
  (FuncNotDefined _ _ x) == (FuncNotDefined _ _ y) = x == y

errorToReport :: ElaborationError -> Report String
errorToReport = \case
  FuncNotDefined pos l n -> let
    msg
      | l == 1 = "there is only one line in the file"
      | otherwise = "there are only " ++ show l ++ " lines in the file"
    in Err Nothing ("function " ++ show n ++ " is not defined") [(pos, This msg)] []

seqEither_ :: ([a] -> b) -> [Either [ElaborationError] a] -> Either [ElaborationError] b
seqEither_ f (partitionEithers -> (es, xs))
  | null es = Right (f xs)
  | otherwise = Left . nub $ concat es

elaborate :: [[SpanInst]] -> Either [Report String] (X7 ())
elaborate fs = let
  fs' = fs <&>
    seqEither_ sequence_ .
    map \(SpanInst x pos) -> deSpan pos <$> case x of
      Pure f -> Right f
      Call n -> case fs' ^? ix (n-1) of
        Just (Right f) -> Right f
        Nothing -> Left [FuncNotDefined pos (length fs) n]
        _ -> Left []
  in seqEither_ (fromMaybe (pure ()) . preview _last) fs' & _Left %~ map errorToReport
