#!/bin/sh -x

rosservice call /move_base/set_logger_level ros.move_base WARN
rosservice call /enemy_bot/move_base/set_logger_level ros.move_base WARN
