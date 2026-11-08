//! Minimal JSON parser (stdlib only) for the parity sweep input file.
//! Supports objects, arrays, strings (with escapes incl. `\uXXXX`),
//! numbers, booleans, and null. Only what `parity.rs` needs.

#[derive(Debug, Clone)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
}

pub struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(format!("trailing characters at byte {}", p.pos));
    }
    Ok(v)
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, s: &str) -> Result<(), String> {
        if self.bytes[self.pos..].starts_with(s.as_bytes()) {
            self.pos += s.len();
            Ok(())
        } else {
            Err(format!("expected {:?} at byte {}", s, self.pos))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => {
                self.expect("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.expect("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.expect("null")?;
                Ok(Json::Null)
            }
            Some(b'-') | Some(b'0'..=b'9') => self.number(),
            Some(c) => Err(format!("unexpected byte {:?} at {}", c as char, self.pos)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect("{")?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(":")?;
            let val = self.value()?;
            pairs.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.pos)),
            }
        }
        Ok(Json::Obj(pairs))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect("[")?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let val = self.value()?;
            items.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.pos)),
            }
        }
        Ok(Json::Arr(items))
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect("\"")?;
        let mut out = String::new();
        loop {
            let b = self.peek().ok_or("unterminated string")?;
            match b {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let e = self.peek().ok_or("unterminated escape")?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = self.hex4()?;
                            if (0xD800..0xDC00).contains(&hex) {
                                // Surrogate pair: expect a following \uDC00-\uDFFF.
                                self.expect("\\u")?;
                                let lo = self.hex4()?;
                                let c = 0x10000 + ((hex - 0xD800) << 10) + (lo - 0xDC00);
                                out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                            } else {
                                out.push(char::from_u32(hex).unwrap_or('\u{FFFD}'));
                            }
                        }
                        _ => {
                            return Err(format!("bad escape \\{} at byte {}", e as char, self.pos))
                        }
                    }
                }
                _ => {
                    // Copy one UTF-8 char.
                    let s =
                        std::str::from_utf8(&self.bytes[self.pos..]).map_err(|e| e.to_string())?;
                    let ch = s.chars().next().ok_or("unterminated string")?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.bytes.len() {
            return Err("truncated \\u escape".to_string());
        }
        let s =
            std::str::from_utf8(&self.bytes[self.pos..self.pos + 4]).map_err(|e| e.to_string())?;
        let v = u32::from_str_radix(s, 16).map_err(|e| e.to_string())?;
        self.pos += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() || b == b'-' || b == b'+' || b == b'.' || b == b'e' || b == b'E' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| e.to_string())?;
        s.parse::<f64>().map(Json::Num).map_err(|e| e.to_string())
    }
}
