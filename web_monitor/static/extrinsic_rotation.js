(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.ExtrinsicRotation = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function multiply3(left, right) {
    var result = new Array(9);
    for (var row = 0; row < 3; row++) {
      for (var column = 0; column < 3; column++) {
        var value = 0;
        for (var index = 0; index < 3; index++) {
          value += left[row * 3 + index] * right[index * 3 + column];
        }
        result[row * 3 + column] = value;
      }
    }
    return result;
  }

  function radians(degrees) {
    return degrees * Math.PI / 180;
  }

  function rotationX(degrees) {
    var angle = radians(degrees);
    var cosine = Math.cos(angle);
    var sine = Math.sin(angle);
    return [1, 0, 0, 0, cosine, -sine, 0, sine, cosine];
  }

  function rotationY(degrees) {
    var angle = radians(degrees);
    var cosine = Math.cos(angle);
    var sine = Math.sin(angle);
    return [cosine, 0, sine, 0, 1, 0, -sine, 0, cosine];
  }

  function rotationZ(degrees) {
    var angle = radians(degrees);
    var cosine = Math.cos(angle);
    var sine = Math.sin(angle);
    return [cosine, -sine, 0, sine, cosine, 0, 0, 0, 1];
  }

  function finiteNumber(value, label) {
    var number = Number(value);
    if (!Number.isFinite(number)) throw new Error(label + " 必须是有限数字");
    return number;
  }

  function clean(value) {
    if (Math.abs(value) < 1e-15) return 0;
    return value;
  }

  var WORLD_ORDERS = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"];

  // order 从左到右是固定世界轴的物理执行顺序；每一步都在世界侧左乘。
  function worldIncrement(rxDegrees, ryDegrees, rzDegrees, order) {
    var rx = finiteNumber(rxDegrees, "世界 X 轴角度");
    var ry = finiteNumber(ryDegrees, "世界 Y 轴角度");
    var rz = finiteNumber(rzDegrees, "世界 Z 轴角度");
    var sequence = order === undefined ? "XYZ" : String(order).toUpperCase();
    if (WORLD_ORDERS.indexOf(sequence) === -1) {
      throw new Error("世界轴旋转顺序必须是 XYZ、XZY、YXZ、YZX、ZXY 或 ZYX");
    }
    var matrices = {X: rotationX(rx), Y: rotationY(ry), Z: rotationZ(rz)};
    var result = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    sequence.split("").forEach(function (axis) {
      result = multiply3(matrices[axis], result);
    });
    return result;
  }

  // T_M_C 是相机 -> 动捕/世界。相机光心固定时，R_new = delta_R_world * R_old，t 不变。
  function applyWorldRotation(extrinsic, rxDegrees, ryDegrees, rzDegrees, order) {
    if (!Array.isArray(extrinsic) || extrinsic.length !== 16) {
      throw new Error("T_M_C 必须包含 16 个数字");
    }
    var result = extrinsic.map(function (value, index) {
      return finiteNumber(value, "T_M_C 第 " + (index + 1) + " 项");
    });
    var oldRotation = [
      result[0], result[1], result[2],
      result[4], result[5], result[6],
      result[8], result[9], result[10]
    ];
    var updated = multiply3(
      worldIncrement(rxDegrees, ryDegrees, rzDegrees, order),
      oldRotation
    );
    var rotationIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10];
    rotationIndices.forEach(function (target, index) {
      result[target] = clean(updated[index]);
    });
    return result;
  }

  return {
    worldOrders: WORLD_ORDERS.slice(),
    multiply3: multiply3,
    worldIncrement: worldIncrement,
    applyWorldRotation: applyWorldRotation
  };
}));
