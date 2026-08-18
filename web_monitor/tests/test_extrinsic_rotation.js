"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const rotation = require("../static/extrinsic_rotation.js");

function assertClose(actual, expected, tolerance = 1e-12) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => {
    assert.ok(
      Math.abs(value - expected[index]) <= tolerance,
      `index ${index}: expected ${expected[index]}, got ${value}`
    );
  });
}

test("identity T_M_C rotated 90 degrees about fixed world X", () => {
  const identity = [
    1, 0, 0, 10,
    0, 1, 0, 20,
    0, 0, 1, 30,
    0, 0, 0, 1
  ];
  const updated = rotation.applyWorldRotation(identity, 90, 0, 0);
  assertClose(updated, [
    1, 0, 0, 10,
    0, 0, -1, 20,
    0, 1, 0, 30,
    0, 0, 0, 1
  ]);
});

test("world-axis increment is left-multiplied and translation stays fixed", () => {
  const extrinsic = [
    1, 0, 0, -1313.5,
    0, 0, -1, -1868.25,
    0, 1, 0, 1061.75,
    0, 0, 0, 1
  ];
  const updated = rotation.applyWorldRotation(extrinsic, 0, 0, 90);
  assertClose(updated, [
    0, 0, 1, -1313.5,
    1, 0, 0, -1868.25,
    0, 1, 0, 1061.75,
    0, 0, 0, 1
  ]);
});

test("fixed world-axis order is X then Y then Z", () => {
  const combined = rotation.worldIncrement(90, 90, 0);
  const expected = rotation.multiply3(
    rotation.worldIncrement(0, 90, 0),
    rotation.worldIncrement(90, 0, 0)
  );
  assertClose(combined, expected);
});

test("all six fixed-world-axis orders follow their visible execution order", () => {
  const degrees = {X:23, Y:-37, Z:61};
  const toRadians = value => value * Math.PI / 180;
  const c = axis => Math.cos(toRadians(degrees[axis]));
  const s = axis => Math.sin(toRadians(degrees[axis]));
  const matrices = {
    X:[1,0,0, 0,c("X"),-s("X"), 0,s("X"),c("X")],
    Y:[c("Y"),0,s("Y"), 0,1,0, -s("Y"),0,c("Y")],
    Z:[c("Z"),-s("Z"),0, s("Z"),c("Z"),0, 0,0,1]
  };
  for (const order of ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]) {
    let expected = [1,0,0, 0,1,0, 0,0,1];
    for (const axis of order) expected = rotation.multiply3(matrices[axis], expected);
    const actual = rotation.worldIncrement(degrees.X, degrees.Y, degrees.Z, order);
    assertClose(actual, expected);
  }
});

test("invalid world-axis order is rejected", () => {
  assert.throws(
    () => rotation.worldIncrement(1, 2, 3, "XXY"),
    /旋转顺序/
  );
});

test("invalid matrix values are rejected", () => {
  assert.throws(
    () => rotation.applyWorldRotation(new Array(16).fill(NaN), 1, 2, 3),
    /有限数字/
  );
});
