# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

def insert_newline(s, n):
    """
    Inserts a newline character after every n characters in the string s.

    Parameters:
    s (str): The input string.
    n (int): The number of characters after which to insert a newline.

    Returns:
    str: The new string with newline characters inserted.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Use list comprehension to insert newline characters
    result = [s[i:i + n] for i in range(0, len(s), n)]

    # Join the list into a single string with newline characters
    return '\n'.join(result)

def last_part(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    if n <= 3:
        return "." * n
    return f"...{s[-(n-3):]}"


def first_part(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    if n <= 3:
        return "." * n
    return f"{s[:(n-3)]}..."


def first_and_last_part(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    if n <= 3:
        return "." * n
    part_len = (n - 3) // 2
    return s[:part_len] + " ... " + s[-part_len:]