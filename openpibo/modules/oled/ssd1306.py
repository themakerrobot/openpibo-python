# The MIT License (MIT)
#
# Copyright (c) 2017 Michael McWethy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
`adafruit_ssd1306`
====================================================
MicroPython SSD1306 OLED driver, I2C and SPI interfaces
* Author(s): Tony DiCola, Michael McWethy
"""

import time

from . import spi_device as spi_device

from . import framebuf as framebuf

__version__ = "0.0.0-auto.0"
__repo__ = "https://github.com/adafruit/Adafruit_CircuitPython_SSD1306.git"

# register definitions
SET_CONTRAST = 0x81
SET_ENTIRE_ON = 0xA4
SET_NORM_INV = 0xA6
SET_DISP = 0xAE
SET_MEM_ADDR = 0x20
SET_COL_ADDR = 0x21
SET_PAGE_ADDR = 0x22
SET_DISP_START_LINE = 0x40
SET_SEG_REMAP = 0xA0
SET_MUX_RATIO = 0xA8
SET_COM_OUT_DIR = 0xC0
SET_DISP_OFFSET = 0xD3
SET_COM_PIN_CFG = 0xDA
SET_DISP_CLK_DIV = 0xD5
SET_PRECHARGE = 0xD9
SET_VCOM_DESEL = 0xDB
SET_CHARGE_PUMP = 0x8D


class _SSD1306(framebuf.FrameBuffer):
  """Base class for SSD1306 display driver"""

  # pylint: disable-msg=too-many-arguments
  def __init__(self, buffer, width, height, *, external_vcc, reset):
    super().__init__(buffer, width, height)
    self.width = width
    self.height = height
    self.external_vcc = external_vcc
    # reset may be None if not needed
    self.reset_pin = reset
    if self.reset_pin:
      self.reset_pin.switch_to_output(value=0)
    self.pages = self.height // 8
    # Note the subclass must initialize self.framebuf to a framebuffer.
    # This is necessary because the underlying data buffer is different
    # between I2C and SPI implementations (I2C needs an extra byte).
    self._power = False
    self.poweron()
    self.init_display()

  @property
  def power(self):
    """True if the display is currently powered on, otherwise False"""
    return self._power

  def init_display(self):
    """Base class to initialize display"""
    for cmd in (
      SET_DISP | 0x00,  # off
      # address setting
      SET_MEM_ADDR,
      0x00,  # horizontal
      # resolution and layout
      SET_DISP_START_LINE | 0x00,
      SET_SEG_REMAP | 0x01,  # column addr 127 mapped to SEG0
      SET_MUX_RATIO,
      self.height - 1,
      SET_COM_OUT_DIR | 0x08,  # scan from COM[N] to COM0
      SET_DISP_OFFSET,
      0x00,
      SET_COM_PIN_CFG,
      0x02 if self.height == 32 or self.height == 16 else 0x12,
      # timing and driving scheme
      SET_DISP_CLK_DIV,
      0x80,
      SET_PRECHARGE,
      0x22 if self.external_vcc else 0xF1,
      SET_VCOM_DESEL,
      0x30,  # 0.83*Vcc
      # display
      SET_CONTRAST,
      0xFF,  # maximum
      SET_ENTIRE_ON,  # output follows RAM contents
      SET_NORM_INV,  # not inverted
      # charge pump
      SET_CHARGE_PUMP,
      0x10 if self.external_vcc else 0x14,
      SET_DISP | 0x01,
    ):  # on
      self.write_cmd(cmd)
    if self.width == 72:
      self.write_cmd(0xAD)
      self.write_cmd(0x30)
    self.fill(0)
    self.show()

  def poweroff(self):
    """Turn off the display (nothing visible)"""
    self.write_cmd(SET_DISP | 0x00)
    self._power = False

  def contrast(self, contrast):
    """Adjust the contrast"""
    self.write_cmd(SET_CONTRAST)
    self.write_cmd(contrast)

  def invert(self, invert):
    """Invert all pixels on the display"""
    self.write_cmd(SET_NORM_INV | (invert & 1))

  def write_framebuf(self):
    """Derived class must implement this"""
    raise NotImplementedError

  def write_cmd(self, cmd):
    """Derived class must implement this"""
    raise NotImplementedError

  def poweron(self):
    "Reset device and turn on the display."
    if self.reset_pin:
      self.reset_pin.value = 1
      time.sleep(0.001)
      self.reset_pin.value = 0
      time.sleep(0.010)
      self.reset_pin.value = 1
      time.sleep(0.010)
    self.write_cmd(SET_DISP | 0x01)
    self._power = True

  def show(self):
    """Update the display"""
    xpos0 = 0
    xpos1 = self.width - 1
    if self.width == 64:
      # displays with width of 64 pixels are shifted by 32
      xpos0 += 32
      xpos1 += 32
    if self.width == 72:
      # displays with width of 72 pixels are shifted by 28
      xpos0 += 28
      xpos1 += 28
    self.write_cmd(SET_COL_ADDR)
    self.write_cmd(xpos0)
    self.write_cmd(xpos1)
    self.write_cmd(SET_PAGE_ADDR)
    self.write_cmd(0)
    self.write_cmd(self.pages - 1)
    self.write_framebuf()


class SSD1306_I2C(_SSD1306):
  """
  I2C class for SSD1306
  :param width: the width of the physical screen in pixels,
  :param height: the height of the physical screen in pixels,
  :param i2c: the I2C peripheral to use,
  :param addr: the 8-bit bus address of the device,
  :param external_vcc: whether external high-voltage source is connected.
  :param reset: if needed, DigitalInOut designating reset pin
  """

  def __init__(
    self, width, height, i2c, *, addr=0x3C, external_vcc=False, reset=None
  ):
    self.i2c_device = i2c_device.I2Device(i2c, addr)
    self.addr = addr
    self.temp = bytearray(2)
    # Add an extra byte to the data buffer to hold an I2C data/command byte
    # to use hardware-compatible I2C transactions.  A memoryview of the
    # buffer is used to mask this byte from the framebuffer operations
    # (without a major memory hit as memoryview doesn't copy to a separate
    # buffer).
    self.buffer = bytearray(((height // 8) * width) + 1)
    self.buffer[0] = 0x40  # Set first byte of data buffer to Co=0, D/C=1
    super().__init__(
      memoryview(self.buffer)[1:],
      width,
      height,
      external_vcc=external_vcc,
      reset=reset,
    )

  def write_cmd(self, cmd):
    """Send a command to the SPI device"""
    self.temp[0] = 0x80  # Co=1, D/C#=0
    self.temp[1] = cmd
    with self.i2c_device:
      self.i2c_device.write(self.temp)

  def write_framebuf(self):
    """Blast out the frame buffer using a single I2C transaction to support
    hardware I2C interfaces."""
    with self.i2c_device:
      self.i2c_device.write(self.buffer)


# pylint: disable-msg=too-many-arguments
class SSD1306_SPI(_SSD1306):
  """
  SPI class for SSD1306
  :param width: the width of the physical screen in pixels,
  :param height: the height of the physical screen in pixels,
  :param spi: the SPI peripheral to use,
  :param dc: the data/command pin to use (often labeled "D/C"),
  :param reset: the reset pin to use,
  :param cs: the chip-select pin to use (sometimes labeled "SS").
  """

  # pylint: disable=no-member
  # Disable should be reconsidered when refactor can be tested.
  def __init__(
    self,
    width,
    height,
    spi,
    dc,
    reset,
    cs,
    *,
    external_vcc=False,
    baudrate=8000000,
    polarity=0,
    phase=0
  ):
    self.rate = 10 * 1024 * 1024
    dc.switch_to_output(value=0)
    self.spi_device = spi_device.SPIDevice(
      spi, cs, baudrate=baudrate, polarity=polarity, phase=phase
    )
    self.dc_pin = dc
    self.buffer = bytearray((height // 8) * width)
    super().__init__(
      memoryview(self.buffer),
      width,
      height,
      external_vcc=external_vcc,
      reset=reset,
    )

  def write_cmd(self, cmd):
    """Send a command to the SPI device"""
    self.dc_pin.value = 0
    with self.spi_device as spi:
      spi.write(bytearray([cmd]))

  def write_framebuf(self):
    """write to the frame buffer via SPI"""
    self.dc_pin.value = 1
    with self.spi_device as spi:
      spi.write(self.buffer)

