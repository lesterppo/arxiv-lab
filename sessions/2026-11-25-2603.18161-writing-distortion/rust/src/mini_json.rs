//! Minimal JSON parser (std only), used ONLY by the harness binaries
//! (`writing_distortion` verify mode and the `parity` bin) to read
//! `sample16.json` and JSON-lines case files. It is not part of the
//! ported metrics; the metrics in `lib.rs` never touch JSON.

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        if let Value::Obj(pairs) = self {
            pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let Value::Num(n) = self {
            Some(*n)
        } else {
            None
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        if let Value::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }
}

pub fn parse(input: &str) -> Result<Value, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut p = Parser { chars: &chars, pos: 0 };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(format!("trailing characters at position {}", p.pos));
    }
    Ok(v)
}

struct Parser<'a> {
    chars: &'a [char],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at position {}", c, self.pos))
        }
    }
    fn value(&mut self) -> Result<Value, String> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => Ok(Value::Str(self.string()?)),
            Some('t') => self.literal("true", Value::Bool(true)),
            Some('f') => self.literal("false", Value::Bool(false)),
            Some('n') => self.literal("null", Value::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(format!("unexpected character '{}' at {}", c, self.pos)),
            None => Err("unexpected end of input".to_string()),
        }
    }
    fn literal(&mut self, word: &str, v: Value) -> Result<Value, String> {
        for wc in word.chars() {
            if self.peek() != Some(wc) {
                return Err(format!("bad literal at position {}", self.pos));
            }
            self.pos += 1;
        }
        Ok(v)
    }
    fn number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        s.parse::<f64>()
            .map(Value::Num)
            .map_err(|_| format!("bad number '{}'", s))
    }
    fn string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err("unterminated string".to_string()),
                Some('"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some('\\') => {
                    self.pos += 1;
                    match self.peek() {
                        Some('"') => {
                            out.push('"');
                            self.pos += 1;
                        }
                        Some('\\') => {
                            out.push('\\');
                            self.pos += 1;
                        }
                        Some('/') => {
                            out.push('/');
                            self.pos += 1;
                        }
                        Some('b') => {
                            out.push('\u{0008}');
                            self.pos += 1;
                        }
                        Some('f') => {
                            out.push('\u{000C}');
                            self.pos += 1;
                        }
                        Some('n') => {
                            out.push('\n');
                            self.pos += 1;
                        }
                        Some('r') => {
                            out.push('\r');
                            self.pos += 1;
                        }
                        Some('t') => {
                            out.push('\t');
                            self.pos += 1;
                        }
                        Some('u') => {
                            self.pos += 1;
                            let cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) {
                                // High surrogate: expect \uXXXX low surrogate.
                                if self.peek() == Some('\\') {
                                    self.pos += 1;
                                    if self.peek() == Some('u') {
                                        self.pos += 1;
                                        let lo = self.hex4()?;
                                        if (0xDC00..0xE000).contains(&lo) {
                                            let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                            out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                                        } else {
                                            out.push('\u{FFFD}');
                                        }
                                    } else {
                                        out.push('\u{FFFD}');
                                    }
                                } else {
                                    out.push('\u{FFFD}');
                                }
                            } else {
                                out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                            }
                        }
                        Some(c) => {
                            return Err(format!("bad escape '\\{}' at {}", c, self.pos));
                        }
                        None => return Err("unterminated string".to_string()),
                    }
                }
                Some(c) => {
                    out.push(c);
                    self.pos += 1;
                }
            }
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        let mut v: u32 = 0;
        for _ in 0..4 {
            match self.peek() {
                Some(c) => {
                    let d = c.to_digit(16).ok_or_else(|| {
                        format!("bad hex digit '{}' at {}", c, self.pos)
                    })?;
                    v = v * 16 + d;
                    self.pos += 1;
                }
                None => return Err("truncated \\u escape".to_string()),
            }
        }
        Ok(v)
    }
    fn array(&mut self) -> Result<Value, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(Value::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.pos += 1;
                }
                Some(']') => {
                    self.pos += 1;
                    return Ok(Value::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.pos)),
            }
        }
    }
    fn object(&mut self) -> Result<Value, String> {
        self.expect('{')?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(Value::Obj(pairs));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(':')?;
            let val = self.value()?;
            pairs.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.pos += 1;
                }
                Some('}') => {
                    self.pos += 1;
                    return Ok(Value::Obj(pairs));
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.pos)),
            }
        }
    }
}
