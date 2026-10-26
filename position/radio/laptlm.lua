-- laptlm.lua : EdgeTX function script (MDL > Special Functions > ON > Lua Script > laptlm)
-- For radios without Lua mix scripts or global variables.
-- Reads stick commands from the laptop over USB serial (SYS > Hardware > USB-VCP = LUA)
-- and writes them to GV1..GV4 (mixer sources) and publishes them as telemetry sensors LRol, LPit, LThr, LYaw (-1024..1024) and LLnk
-- (1 = laptop sending, 0 = lost). Use the sensors as mixer sources behind switch SA.
-- Line formats from laptop, values -1024..1024:
--   "C<roll>,<pitch>,<thr>,<yaw>\n"            stick commands
--   "P<px>,<py>,<pz>,<vx>,<vy>,<vz>,<ok>\n"     room position error and velocity for the RAPTOR policy
--     -> sensors LPx LPy LPz LVx LVy LVz LOk, used as Inputs for CH9..CH15. LOk is -1024 without a fix,
--        otherwise it alternates 1024 / 512 with every run that got new data, so the FC can tell it is still coming.
-- Sends a status line back to the laptop 5x a second (arm/mode/laptop switch + FM telemetry).
-- Beeps: low tone once when the script starts, high tone when first laptop data arrives.
-- If no valid line for 300 ms, the sensors follow the pilot's own sticks (safe fallback).

local TIMEOUT = 30          -- getTime() ticks are 10 ms -> 300 ms
local ID = 0x5A00           -- private sensor IDs, one per value
local NAMES = { "LRol", "LPit", "LThr", "LYaw" }
local PNAMES = { "LPx", "LPy", "LPz", "LVx", "LVy", "LVz", "LOk" }   -- sensor ids ID+5 .. ID+11
local PTIMEOUT = 25         -- 250 ms without a position line -> not valid
local pvals = { 0, 0, 0, 0, 0, 0, -1024 }
local lastP = -1000
local beat = false
local newP = false          -- a position line arrived during this run
local buf = ""
local vals = { 0, 0, -1024, 0 }
local lastRx = -1000
local started = false
local gotData = false
local rxBytes = 0
local lastEcho = 0
local lastGv = {}
local lastFm = -1

local function clamp(v)
  v = tonumber(v)
  if v > 1024 then return 1024 end
  if v < -1024 then return -1024 end
  return v
end

local function parse(line)
  if string.sub(line, 1, 1) == "P" then
    local a, b, c, d, e, f, ok = string.match(line, "^P(%-?%d+),(%-?%d+),(%-?%d+),(%-?%d+),(%-?%d+),(%-?%d+),(%-?%d+)")
    if a then
      pvals = { clamp(a), clamp(b), clamp(c), clamp(d), clamp(e), clamp(f), clamp(ok) }
      lastP = getTime()
      newP = true
    end
    return
  end
  local r, p, t, y = string.match(line, "^C(%-?%d+),(%-?%d+),(%-?%d+),(%-?%d+)")
  if r then
    vals[1], vals[2], vals[3], vals[4] = clamp(r), clamp(p), clamp(t), clamp(y)
    lastRx = getTime()
    if not gotData then gotData = true; playTone(2000, 150, 100) end
  end
end

local function publish(i, v, name)
  -- id, subId, instance, value, unit (0 = raw), precision, name
  setTelemetryValue(ID + i, 0, 0, math.floor(v), 0, 0, name)
end

local function run(event)
  if not started then started = true; playTone(800, 300, 0) end
  -- serialRead() without a size stops at the first newline, so one call per run handled one line
  -- per run (~22 a second) and the rest queued up. Drain everything that has arrived instead.
  for _ = 1, 16 do
    local data = serialRead(64)
    if not data or #data == 0 then break end
    rxBytes = rxBytes + #data
    buf = buf .. data
  end
  if #buf > 0 then
    if #buf > 512 then buf = string.sub(buf, -256) end
    local nl = string.find(buf, "\n", 1, true)
    while nl do
      parse(string.sub(buf, 1, nl - 1))
      buf = string.sub(buf, nl + 1)
      nl = string.find(buf, "\n", 1, true)
    end
  end

  -- 5x a second report back to the laptop:
  -- "S<bytes received>,<ch5 arm>,<ch7 mode>,<ch8 laptop switch>,<FM telemetry>,<ch3 throttle out>,<throttle stick>,<ch1 out>,<ch2 out>,<ch4 out>,<ch6 RAPTOR switch>,<ch9 out>,<ch15 out>"
  -- FM is Betaflight's flight mode over ELRS (e.g. ANGL, ACRO; "*" appended = disarmed), "-" if none
  if getTime() - lastEcho > 20 then
    lastEcho = getTime()
    local fm = getValue("FM")
    if type(fm) ~= "string" or fm == "" then fm = "-" end
    serialWrite(string.format("S%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d\n", rxBytes,
      getValue("ch5"), getValue("ch7"), getValue("ch8"), fm, getValue("ch3"), getValue("thr"),
      getValue("ch1"), getValue("ch2"), getValue("ch4"), getValue("ch6"), getValue("ch9"), getValue("ch15")))
  end

  local out
  if getTime() - lastRx < TIMEOUT then
    out = vals
    publish(4, 1, "LLnk")
  else
    out = { getValue("ail"), getValue("ele"), getValue("thr"), getValue("rud") }
    publish(4, 0, "LLnk")
  end
  -- position for the policy: zeros and "no fix" when the laptop stopped sending positions
  if newP then beat = not beat; newP = false end   -- flip once per run, whatever the number of lines
  if getTime() - lastP < PTIMEOUT and pvals[7] > 0 then
    for i = 1, 6 do publish(4 + i, pvals[i], PNAMES[i]) end
    publish(11, beat and 1024 or 512, PNAMES[7])
  else
    for i = 1, 6 do publish(4 + i, 0, PNAMES[i]) end
    publish(11, -1024, PNAMES[7])
  end

  local fm = getFlightMode()
  for i = 1, 4 do
    publish(i - 1, out[i], NAMES[i])
    -- GV1..GV4 are the mixer sources: CH1..CH4 line "GVn, SA down, Replace".
    -- GVs are stored per flight mode, so write to the flight mode that is active right now.
    local v = math.floor(out[i])
    if v ~= lastGv[i] or fm ~= lastFm then
      model.setGlobalVariable(i - 1, fm, v)
      lastGv[i] = v
    end
  end
  lastFm = fm
end

return { run = run }
