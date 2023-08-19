module Parser where

import Control.Applicative (liftA2)
import Control.Monad
import Control.Lens
import Data.Char
import Data.Either
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
  <|> o '+' (op2 drillAtom $ through2' _Rat (+))
  <?> "an instruction"

func :: Parser [SpanInst]
func = hspace >> many do
  SourcePos f row1 col1 <- getSourcePos
  i <- inst
  SourcePos _ row2 col2 <- getSourcePos
  hspace
  pure . SpanInst i $ Position (unPos row1, unPos col1) (unPos row2, unPos col2) f

x7 :: Parser [[SpanInst]]
x7 = [] <$ eof <|> liftA2 (:) (func <* (void eol <|> eof)) x7

seqEither_ :: ([a] -> b) -> [Either [e] a] -> Either [e] b
seqEither_ f (partitionEithers -> (es, xs))
  | null es = Right (f xs)
  | otherwise = Left (concat es)

elaborate :: [[SpanInst]] -> Either [Report String] (X7 ())
elaborate fs = let
  msg
    | length fs == 1 = "there is only one line in the file"
    | otherwise = "there are only " ++ show (length fs) ++ " lines in the file"
  fs' = fs <&>
    seqEither_ sequence_ .
    map \(SpanInst x pos) -> deSpan pos <$> case x of
      Pure f -> Right f
      Call n -> case fs' ^? ix n of
        Just (Right f) -> Right f
        Nothing -> Left
          [Err Nothing ("function " ++ show n ++ " is not defined")
               [(pos, This msg)] []]
        _ -> Left []
  in seqEither_ last fs'
