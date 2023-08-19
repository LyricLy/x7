module Main where

import Data.List
import Data.Void
import Error.Diagnose
import Error.Diagnose.Compat.Megaparsec
import System.Exit
import Options.Applicative
import Text.Megaparsec
import State
import Parser hiding (Parser)

instance HasHints Void a where
  hints _ = mempty

data Opts = Opts String

opts' :: Parser Opts
opts' = Opts
  <$> argument str (metavar "FILE")

opts :: ParserInfo Opts
opts = info (opts' <**> helper) (fullDesc <> progDesc "Execute FILE" <> header "x7, the exceptional golfing language")

printDiag :: Diagnostic String -> IO ()
printDiag = printDiagnostic stderr WithUnicode (TabSize 4) defaultStyle

main :: IO ()
main = do
  Opts filename <- execParser opts
  t <- readFile filename
  rfs <- case runParser x7 filename t of
    Left bundle ->
      let diag = errorDiagnosticFromBundle Nothing "parsing failed" Nothing bundle
          diag' = addFile diag filename t
      in printDiag diag' >> exitWith (ExitFailure 2)
    Right rfs -> pure rfs
  let theDiagnostic = addFile mempty filename t
  code <- case elaborate rfs of
    Left reports -> printDiag (foldl' addReport theDiagnostic reports) >> exitWith (ExitFailure 2)
    Right c -> pure c
  case runX7 code of
    Left r -> printDiag (addReport theDiagnostic (reportRaise r)) >> exitWith (ExitFailure 1)
    Right x -> print x
